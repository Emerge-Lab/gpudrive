import torch
import torch.nn.functional as F
import Quantizer
import json
import os
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Geometry:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Geometry(x={self.x}, y={self.y}, z={self.z})"

def get_geometries(json_file_path):
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"File not found: {json_file_path}")
    if os.path.getsize(json_file_path) == 0:
        raise ValueError(f"File is empty: {json_file_path}")
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    roads = data.get("roads", [])

    geometries = []
    for road in roads:
        road_i_geometry = []
        id_i = road.get("id", "")
        for geometry in road.get("geometry", []):
            x = geometry.get("x", 0.0)
            y = geometry.get("y", 0.0)
            road_i_geometry.append(Geometry(x, y))
        geometries.append([id_i, road_i_geometry])

    return geometries

def geometries_fake_quantization_file(json_file_path, Qmin=-128, Qmax=127):
    '''Replaces all the geometries in the json file with their fake-quantized versions'''

    geometries = get_geometries(json_file_path)
    print(f'Number of road objects: {len(geometries)}')
    for road_i, road in enumerate(geometries):
        print(f'Processing road with id: {road[0]}/{len(geometries)}')
        x = torch.tensor([g.x for g in road[1]], device=device, dtype=torch.float32)
        y = torch.tensor([g.y for g in road[1]], device=device, dtype=torch.float32)

        x_hat = fake_quantization(x, Qmin=Qmin, Qmax=Qmax)
        y_hat = fake_quantization(y, Qmin=Qmin, Qmax=Qmax)

        print(f"Final MSE loss x: {F.mse_loss(x, x_hat).item()}")
        print(f"Final MSE loss y: {F.mse_loss(y, y_hat).item()}")

        for i, g in enumerate(road[1]):
            g.x = x_hat[i].item()
            g.y = y_hat[i].item()
    
    # Save the modified geometries back to same json file
    # Read the data first
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    for road_idx, road in enumerate(data.get("roads", [])):
        for geom_idx, geometry in enumerate(road.get("geometry", [])):
            geometry["x"] = geometries[road_idx][1][geom_idx].x
            geometry["y"] = geometries[road_idx][1][geom_idx].y
    # Now write the modified data back
    with open(json_file_path, 'w') as f:
        json.dump(data, f)

def fake_quantization(x: torch.tensor, Qmin=-128, Qmax=127, train_logs=True):
    normalizer = Quantizer.Standardizer(Qmin=Qmin, Qmax=Qmax)
    normalizer.fit(x)

    # Scale the input tensor
    # x_scale = Quantizer.scale_mapper_func(x, const=1.0)
    x_scale = normalizer.transform(x)

    # Calculate the min and max of the scale mapped float values
    fmin, fmax = x_scale.min().item(), x_scale.max().item()

    # Parameters: scale (c) and offset (d)
    # Initialize as learnable parameters
    c = torch.nn.Parameter(torch.tensor((fmax-fmin)/(Qmax-Qmin), device=device))  # scale
    d = torch.nn.Parameter(torch.tensor((fmax*Qmin-fmin*Qmax)/(Qmax-Qmin), device=device))  # zero-point
    Quantizer.train(x_scale, c, d, Qmin=Qmin, Qmax=Qmax, train_logs=train_logs)
    if c.item() != 0.0:
        x_q_final = torch.round((x_scale - d) / c).clamp(Qmin, Qmax)
    else:
        x_q_final = torch.tensor([Qmax]*len(x_scale), device=device)
    x_hat_scale = c * x_q_final + d
    x_hat_final = normalizer.inverse_transform(x_hat_scale)

    return x_hat_final

class QuantizerAnalysis:
    def __init__(self, x_mse, y_mse, road_id, road_length, scenario_file_name, scenario_id = None):
        self.x_mse = x_mse
        self.y_mse = y_mse
        self.road_id = road_id
        self.road_length = road_length
        self.scenario_id = scenario_id
        self.scenario_file_name = scenario_file_name
    


def geometries_fake_quantization(json_file_dir, json_file_dir_quantized, Qmin=-128, Qmax=127, save_files=True, train_logs=True):

    if save_files:
        if not os.path.exists(json_file_dir_quantized):
            os.makedirs(json_file_dir_quantized)

    # Create Analyzer
    analyzer = []
    start_time = time.time()
    json_files = [f for f in os.listdir(json_file_dir) if f.endswith('.json')]
    for json_file_i,json_file in enumerate(json_files):
        i_start_time = time.time()
        json_file_path = os.path.join(json_file_dir, json_file)
        geometries = get_geometries(json_file_path)
        print(f'Number of road objects in {json_file_i+1}th file: {len(geometries)}')
        for road_i, road in enumerate(geometries):
            # print(f'Processing road with id: {road[0]}/{len(geometries)}')
            x = torch.tensor([g.x for g in road[1]], device=device, dtype=torch.float32)
            y = torch.tensor([g.y for g in road[1]], device=device, dtype=torch.float32)

            x_hat = fake_quantization(x, Qmin=Qmin, Qmax=Qmax, train_logs=train_logs)
            y_hat = fake_quantization(y, Qmin=Qmin, Qmax=Qmax, train_logs=train_logs)

            x_mse = F.mse_loss(x, x_hat).item()
            y_mse = F.mse_loss(y, y_hat).item()
            # print(f"Final MSE loss x: {x_mse}")
            # print(f"Final MSE loss y: {y_mse}")

            analyzer.append(QuantizerAnalysis(
                x_mse=x_mse,
                y_mse=y_mse,
                road_id=road[0],
                road_length=len(road[1]),
                scenario_file_name=json_file
            ))

            for i, g in enumerate(road[1]):
                g.x = x_hat[i].item()
                g.y = y_hat[i].item()
        
        # Save the modified geometries to quantized json file
        if save_files:
            with open(json_file_path, 'r') as f:
                data = json.load(f)

            for road_idx, road in enumerate(data.get("roads", [])):
                for geom_idx, geometry in enumerate(road.get("geometry", [])):
                    geometry["x"] = geometries[road_idx][1][geom_idx].x
                    geometry["y"] = geometries[road_idx][1][geom_idx].y

            with open(os.path.join(json_file_dir_quantized, json_file), 'w') as f:
                json.dump(data, f)
        
        i_end_time = time.time()
        print(f"Processed {len(geometries)} roads in {i_end_time - i_start_time:.2f} seconds for file: {json_file}")
    
    end_time = time.time()
    print(f"Processed {len(json_files)} files in {end_time - start_time:.2f} seconds")
    # Save analyzer data to a .csv file
    analyzer_file_path = '/scratch/pm3881/gpudrive/quantizedMapPOC/quantizer_analysis.csv'
    with open(analyzer_file_path, 'w') as f:
        f.write("road_id,road_length,x_mse,y_mse,scenario_file_name\n")
        for analysis in analyzer:
            f.write(f"{analysis.road_id},{analysis.road_length},{analysis.x_mse},{analysis.y_mse},{analysis.scenario_file_name}\n")
    
    # Print basic statistics for analyzer
    print(f"Total roads processed: {len(analyzer)}")
    # print avg x_mse and y_mse
    avg_x_mse = 0.0
    avg_y_mse = 0.0
    if len(analyzer) > 0:
        avg_x_mse = sum(a.x_mse for a in analyzer) / len(analyzer)
        avg_y_mse = sum(a.y_mse for a in analyzer) / len(analyzer)
        print(f"Average x MSE: {avg_x_mse}")
        print(f"Average y MSE: {avg_y_mse}")

    # Std of x_mse and y_mse
    std_x_mse = 0.0
    std_y_mse = 0.0
    if len(analyzer) > 0:
        std_x_mse = (sum((a.x_mse - avg_x_mse) ** 2 for a in analyzer) / len(analyzer)) ** 0.5
        std_y_mse = (sum((a.y_mse - avg_y_mse) ** 2 for a in analyzer) / len(analyzer)) ** 0.5
    print(f"Std of x MSE: {std_x_mse}")
    print(f"Std of y MSE: {std_y_mse}")

if __name__ == "__main__":
    geometries = get_geometries('/scratch/pm3881/gpudrive/data/processed/tl/tfrecord-00000-of-00150_2a173b334be3d32c.json')
    print(f'Number of road objects: {len(geometries)}')

    geom = [g[1] for g in geometries if g[0] == 267][0]
    print(geom)

    # Sample data (your float list)
    g_x = torch.tensor([g.x for g in geom], device=device, dtype=torch.float32)
    g_y = torch.tensor([g.y for g in geom], device=device, dtype=torch.float32)

    # Quantization bounds (e.g., int8: 0 to 255 or -128 to 127)
    Qmin, Qmax = -128, 127

    x = g_y

    normalizer = Quantizer.Standardizer(Qmin=Qmin, Qmax=Qmax)
    normalizer.fit(x)

    # Scale the input tensor
    # x_scale = Quantizer.scale_mapper_func(x, const=1.0)
    x_scale = normalizer.transform(x)

    # Bounds of float values
    fmin_org, fmax_org = x.min().item(), x.max().item()

    # Calculate the min and max of the scale mapped float values
    fmin, fmax = x_scale.min().item(), x_scale.max().item()

    # Parameters: scale (c) and offset (d)
    # Initialize as learnable parameters
    c = torch.nn.Parameter(torch.tensor((fmax_org-fmin_org)/(Qmax-Qmin), device=device))  # scale
    d = torch.nn.Parameter(torch.tensor((fmax_org*Qmin-fmin_org*Qmax)/(Qmax-Qmin), device=device))  # zero-point

    # Cheat Inits
    # c = torch.nn.Parameter(torch.tensor(0.01, device=device))
    # d = torch.nn.Parameter(torch.tensor(0.02, device=device))

    print("Original values:", x)
    print("Scaled values:", x_scale)
    print(f'Max: {fmax}, Min: {fmin}')

    # Strategy 1: Without initial scaling
    print("Strategy 1: Without initial scaling")
    Quantizer.train(x, c, d, Qmin=Qmin, Qmax=Qmax, train_logs=False)
    x_q_final = torch.round((x - d) / c).clamp(Qmin, Qmax)
    x_hat_final = c * x_q_final + d

    print("Final quantized values:", x_q_final)
    print("Final dequantized values:", x_hat_final)
    print(f"Final MSE loss: {F.mse_loss(x, x_hat_final).item()}")
    print("\n\n\n")

    # Strategy 2: With initial scaling
    print("Strategy 2: With initial scaling")
    c = torch.nn.Parameter(torch.tensor((fmax-fmin)/(Qmax-Qmin), device=device))  # scale
    d = torch.nn.Parameter(torch.tensor((fmax*Qmin-fmin*Qmax)/(Qmax-Qmin), device=device))  # zero-point
    print(f'Initial parameters: c={c.item()}, d={d.item()}')
    Quantizer.train(x_scale, c, d, Qmin=Qmin, Qmax=Qmax, train_logs=False)
    print(f"Trained parameters: c={c.item()}, d={d.item()}")
    if c.item() != 0.0:
        x_q_final = torch.round((x_scale - d) / c).clamp(Qmin, Qmax)
    else:
        x_q_final = torch.tensor([Qmax]*len(x_scale), device=device)
    x_hat_scale = c * x_q_final + d
    x_hat_final = normalizer.inverse_transform(x_hat_scale)

    print("Scaled values:", x_scale)
    print("Final quantized values:", x_q_final)
    print("Final dequantized values:", x_hat_final)
    print(f"Final MSE loss: {F.mse_loss(x, x_hat_final).item()}")
    print("\n\n\n")
    # print(softmax_contrastive_loss(x, x_comp).item())
    # x_neg_comp = x_comp.clone()
    # x_neg_comp[0] = -x_neg_comp[0]
    # print(f'Contrastive loss for {x_neg_comp} = {softmax_contrastive_loss(x, x_neg_comp)}')

    # print(f'MSE loss for {x_comp} = {F.mse_loss(x, x_comp).item()}')
    # print(f'MSE loss for {x_neg_comp} = {F.mse_loss(x, x_neg_comp).item()}')


    # geometries_fake_quantization('data/processed/tl/tfrecord-00000-of-00150_2a173b334be3d32c.json', Qmin=-128, Qmax=127)
    geometries_fake_quantization(json_file_dir='/scratch/pm3881/gpudrive/data/processed/tl', json_file_dir_quantized='/scratch/pm3881/gpudrive/data/processed/tl_fake_quantized', Qmin=-128, Qmax=127, save_files=False, train_logs=False)
