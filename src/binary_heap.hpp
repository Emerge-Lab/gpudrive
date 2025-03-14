/*
 *
 * Copyright (c) 1994
 * Hewlett-Packard Company
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  Hewlett-Packard Company makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 *
 * Copyright (c) 1997
 * Silicon Graphics Computer Systems, Inc.
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  Silicon Graphics makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 */

/* NOTE: This is an internal header file, included by other STL headers.
 *   You should not attempt to use it directly.
 */
#pragma once

// Heap-manipulation functions: push_heap, pop_heap, make_heap, sort_heap.

template <class _RandomAccessIterator, class _Distance, class _Tp,
          class _Compare>

inline void __push_heap(_RandomAccessIterator __first, _Distance __holeIndex,
                        _Distance __topIndex, _Tp __x, _Compare __comp) {
  _Distance __parent = (__holeIndex - 1) / 2;
  while (__holeIndex > __topIndex && __comp(*(__first + __parent), __x)) {
    *(__first + __holeIndex) = *(__first + __parent);
    __holeIndex = __parent;
    __parent = (__holeIndex - 1) / 2;
  }
  *(__first + __holeIndex) = __x;
}

template <class _RandomAccessIterator, class _Compare>

inline void __push_heap_aux(_RandomAccessIterator __first,
                            _RandomAccessIterator __last, _Compare __comp) {
  int hole_index = (__last - __first) - 1;
  int top_index = 0;
  __push_heap(__first, hole_index, top_index, *(__last - 1), __comp);
}

template <class _RandomAccessIterator, class _Compare>

inline void push_heap(_RandomAccessIterator __first,
                      _RandomAccessIterator __last, _Compare __comp) {
  __push_heap_aux(__first, __last, __comp);
}

template <class _RandomAccessIterator, class _Distance, class _Tp>

void __adjust_heap(_RandomAccessIterator __first, _Distance __holeIndex,
                   _Distance __len, _Tp __x) {
  _Distance __topIndex = __holeIndex;
  _Distance __secondChild = 2 * __holeIndex + 2;
  while (__secondChild < __len) {
    if (*(__first + __secondChild) < *(__first + (__secondChild - 1)))
      __secondChild--;
    *(__first + __holeIndex) = *(__first + __secondChild);
    __holeIndex = __secondChild;
    __secondChild = 2 * (__secondChild + 1);
  }
  if (__secondChild == __len) {
    *(__first + __holeIndex) = *(__first + (__secondChild - 1));
    __holeIndex = __secondChild - 1;
  }
  __push_heap(__first, __holeIndex, __topIndex, __x);
}

template <class _RandomAccessIterator, class _Tp>

inline void __pop_heap(_RandomAccessIterator __first,
                       _RandomAccessIterator __last,
                       _RandomAccessIterator __result, _Tp __x) {
  *__result = *__first;
  __adjust_heap(__first, 0, __last - __first, __x);
}

template <class _RandomAccessIterator>

inline void __pop_heap_aux(_RandomAccessIterator __first,
                           _RandomAccessIterator __last) {
  __pop_heap(__first, __last - 1, __last - 1, *(__last - 1));
}

template <class _RandomAccessIterator>

inline void pop_heap(_RandomAccessIterator __first,
                     _RandomAccessIterator __last) {
  __pop_heap_aux(__first, __last, __VALUE_TYPE(__first));
}

template <class _RandomAccessIterator, class _Distance, class _Tp,
          class _Compare>

void __adjust_heap(_RandomAccessIterator __first, _Distance __holeIndex,
                   _Distance __len, _Tp __x, _Compare __comp) {
  _Distance __topIndex = __holeIndex;
  _Distance __secondChild = 2 * __holeIndex + 2;
  while (__secondChild < __len) {
    if (__comp(*(__first + __secondChild), *(__first + (__secondChild - 1))))
      __secondChild--;
    *(__first + __holeIndex) = *(__first + __secondChild);
    __holeIndex = __secondChild;
    __secondChild = 2 * (__secondChild + 1);
  }
  if (__secondChild == __len) {
    *(__first + __holeIndex) = *(__first + (__secondChild - 1));
    __holeIndex = __secondChild - 1;
  }
  __push_heap(__first, __holeIndex, __topIndex, __x, __comp);
}

template <class _RandomAccessIterator, class _Tp, class _Compare>

inline void
__pop_heap(_RandomAccessIterator __first, _RandomAccessIterator __last,
           _RandomAccessIterator __result, _Tp __x, _Compare __comp) {
  *__result = *__first;

  int hole_index = 0;
  int len = __last - __first;

  __adjust_heap(__first, hole_index, len, __x, __comp);
}

template <class _RandomAccessIterator, class _Compare>

inline void __pop_heap_aux(_RandomAccessIterator __first,
                           _RandomAccessIterator __last, _Compare __comp) {
  __pop_heap(__first, __last - 1, __last - 1, *(__last - 1), __comp);
}

template <class _RandomAccessIterator, class _Compare>

inline void pop_heap(_RandomAccessIterator __first,
                     _RandomAccessIterator __last, _Compare __comp) {
  __pop_heap_aux(__first, __last, __comp);
}

template <class _RandomAccessIterator>

void __make_heap(_RandomAccessIterator __first, _RandomAccessIterator __last) {
  if (__last - __first < 2)
    return;
  int __len = __last - __first;
  int __parent = (__len - 2) / 2;

  while (true) {
    __adjust_heap(__first, __parent, __len, *(__first + __parent));
    if (__parent == 0)
      return;
    __parent--;
  }
}

template <class _RandomAccessIterator, class _Compare>

void __make_heap(_RandomAccessIterator __first, _RandomAccessIterator __last,
                 _Compare __comp) {
  if (__last - __first < 2)
    return;
  int __len = __last - __first;
  int __parent = (__len - 2) / 2;

  while (true) {
    __adjust_heap(__first, __parent, __len, *(__first + __parent), __comp);
    if (__parent == 0)
      return;
    __parent--;
  }
}

template <class _RandomAccessIterator, class _Compare>

inline void make_heap(_RandomAccessIterator __first,
                      _RandomAccessIterator __last, _Compare __comp) {
  __make_heap(__first, __last, __comp);
}
