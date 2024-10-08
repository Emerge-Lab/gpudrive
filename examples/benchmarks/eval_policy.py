import torch
from pygpudrive.env.config import (
    EnvConfig,
    RenderConfig,
    SceneConfig,
)
from pygpudrive.env.env_torch import GPUDriveTorchEnv
from pygpudrive.agents.policy_actor import PolicyActor
from pygpudrive.agents.core import merge_actions
import mediapy

if __name__ == "__main__":

    # Constants
    MAX_CONTROLLED_AGENTS = 128 
    NUM_WORLDS = 200
    DEVICE = "cuda"
    DATA_PATH = "data/formatted_json_v2_no_tl_train_processed"
    TRAINED_POLICY_PATH = "models/learned_sb3_policy.zip" # Pre-trained policy on 1000 scenes
    SAVE_FILE_PATH = f"videos/project_page/"
    FPS = 20

    # Configs
    render_config = RenderConfig(draw_obj_idx=True)
    scene_config = SceneConfig(path=DATA_PATH, num_scenes=NUM_WORLDS)
    env_config = EnvConfig(
        dynamics_model="classic",
        reward_type="weighted_combination",
        collision_weight=-.1,
        off_road_weight=-.1,
    )
    
    # Make environment
    env = GPUDriveTorchEnv(
        config=env_config,
        scene_config=scene_config,
        render_config=render_config,
        max_cont_agents=MAX_CONTROLLED_AGENTS,  # Maximum number of agents to control per scene
        device=DEVICE,
    )
    
    # Create sim agent
    obj_idx = torch.arange(env_config.max_num_agents_in_scene)

    policy_actor = PolicyActor(
        is_controlled_func=(obj_idx >= 0),
        valid_agent_mask=env.cont_agent_mask,
        saved_model_path=TRAINED_POLICY_PATH,
        device=DEVICE,
        deterministic=True,
    )

    obs = env.reset()

    frames_dict = {f"scene_{idx}": [] for idx in range(NUM_WORLDS)}

    # STEP THROUGH ENVIRONMENT
    for time_step in range(env_config.episode_len):
        print(f"Step {time_step}/{env_config.episode_len}")

        # SELECT ACTIONS
        policy_actions = policy_actor.select_action(obs)
        
        actions = merge_actions(
            actor_actions_dict={"policy": policy_actions},
            actor_ids_dict={"policy": policy_actor.actor_ids},
            reference_action_tensor=env.cont_agent_mask,
            device=DEVICE,
        )
        
        # STEP
        env.step_dynamics(actions)

        # GET NEXT OBS
        obs = env.get_obs()
        dones = env.get_dones()

        # RENDER
        for env_idx in range(NUM_WORLDS):
            if dones[env_idx].all():
                continue
            else:
                frame = env.render(
                    world_render_idx=env_idx,
                    color_objects_by_actor={
                        "policy": policy_actor.actor_ids[env_idx].tolist(),
                    },
                )
                frames_dict[f"scene_{env_idx}"].append(frame)
                
 
    # # # # # # # #
    # Done. Save videos and snapshots
    for scene_name, frames_list in frames_dict.items():
        save_video_path = f"{SAVE_FILE_PATH}/videos/{scene_name}.mp4"
        save_img_path = f"{SAVE_FILE_PATH}/images/{scene_name}.png"
        
        print(f"Saving video to {save_video_path}")

        mediapy.write_video(save_video_path, frames_list, fps=FPS)
        mediapy.write_image(save_img_path, frames_list[10])
        
