//
// distance.h — OPTIMIZED: aligned AVX loads + FMA fused multiply-add
// Changes:
//   1. _mm256_loadu_ps → _mm256_load_ps (32B aligned, skip alignment penalty)
//   2. sub+mul+add → sub + _mm256_fmadd_ps (3 insns vs 4, halves mul+add latency)
//   3. SSE path also upgraded to _mm_load_ps (already aligned) + FMA

#ifndef EFANNA2E_DISTANCE_H
#define EFANNA2E_DISTANCE_H

#include <x86intrin.h>
#include <iostream>
namespace efanna2e{
  enum Metric{
    L2 = 0,
    INNER_PRODUCT = 1,
    FAST_L2 = 2,
    PQ = 3
  };
    class Distance {
    public:
        virtual float compare(const float* a, const float* b, unsigned length) const = 0;
        virtual ~Distance() {}
    };

    class DistanceL2 : public Distance{
    public:
        float compare(const float* a, const float* b, unsigned size) const {
            float result = 0;

#ifdef __GNUC__
#ifdef __AVX__
            // ── AVX2 + FMA: aligned loads, fused multiply-add ──
            //   C = A² - B²  →  diff = A - B; sum += diff * diff
            //   Before: loadu + loadu + sub + mul + add = 5 ops
            //   After:  load  + load  + sub + fmadd    = 4 ops (+ aligned)

  #define AVX_L2SQR_FMA(addr1, addr2, dest, tmp1) \
      tmp1 = _mm256_load_ps(addr1);              \
      tmp1 = _mm256_sub_ps(tmp1, _mm256_load_ps(addr2)); \
      dest = _mm256_fmadd_ps(tmp1, tmp1, dest);

      __m256 sum;
      __m256 l0;
      unsigned D = (size + 7) & ~7U;
      unsigned DR = D % 16;
      unsigned DD = D - DR;
      const float *l = a;
      const float *r = b;
      const float *e_l = l + DD;
      const float *e_r = r + DD;
      float unpack[8] __attribute__ ((aligned (32))) = {0};

      sum = _mm256_load_ps(unpack);
      if(DR){ AVX_L2SQR_FMA(e_l, e_r, sum, l0); }

      for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
        AVX_L2SQR_FMA(l,     r,     sum, l0);
        AVX_L2SQR_FMA(l + 8, r + 8, sum, l0);
      }
      _mm256_store_ps(unpack, sum);
      result = unpack[0] + unpack[1] + unpack[2] + unpack[3]
             + unpack[4] + unpack[5] + unpack[6] + unpack[7];

#else
#ifdef __SSE2__
            // ── SSE4.1 + FMA: aligned loads, fused multiply-add ──
  #define SSE_L2SQR_FMA(addr1, addr2, dest, tmp1) \
          tmp1 = _mm_load_ps(addr1);             \
          tmp1 = _mm_sub_ps(tmp1, _mm_load_ps(addr2)); \
          dest = _mm_fmadd_ps(tmp1, tmp1, dest);

  __m128 sum;
  __m128 l0;
  unsigned D = (size + 3) & ~3U;
  unsigned DR = D % 16;
  unsigned DD = D - DR;
  const float *l = a;
  const float *r = b;
  const float *e_l = l + DD;
  const float *e_r = r + DD;
  float unpack[4] __attribute__ ((aligned (16))) = {0};

  sum = _mm_load_ps(unpack);
  switch (DR) {
      case 12: SSE_L2SQR_FMA(e_l+8, e_r+8, sum, l0);
      case 8:  SSE_L2SQR_FMA(e_l+4, e_r+4, sum, l0);
      case 4:  SSE_L2SQR_FMA(e_l, e_r, sum, l0);
    default: break;
  }
  for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
      SSE_L2SQR_FMA(l,     r,     sum, l0);
      SSE_L2SQR_FMA(l + 4, r + 4, sum, l0);
      SSE_L2SQR_FMA(l + 8, r + 8, sum, l0);
      SSE_L2SQR_FMA(l + 12,r + 12,sum, l0);
  }
  _mm_store_ps(unpack, sum);
  result += unpack[0] + unpack[1] + unpack[2] + unpack[3];

// fallback: scalar
#else
      float diff0, diff1, diff2, diff3;
      const float* last = a + size;
      const float* unroll_group = last - 3;
      while (a < unroll_group) {
          diff0 = a[0] - b[0];
          diff1 = a[1] - b[1];
          diff2 = a[2] - b[2];
          diff3 = a[3] - b[3];
          result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
          a += 4; b += 4;
      }
      while (a < last) {
          diff0 = *a++ - *b++;
          result += diff0 * diff0;
      }
#endif
#endif
#endif
            return result;
        }
    };

  class DistanceInnerProduct : public Distance{
  public:
    float compare(const float* a, const float* b, unsigned size) const {
      float result = 0;
#ifdef __GNUC__
#ifdef __AVX__
      // ── AVX2 + FMA inner product ──
      #define AVX_DOT_FMA(addr1, addr2, dest, tmp1) \
          tmp1 = _mm256_load_ps(addr1); \
          dest = _mm256_fmadd_ps(tmp1, _mm256_load_ps(addr2), dest);

      __m256 sum;
      __m256 l0;
      unsigned D = (size + 7) & ~7U;
      unsigned DR = D % 16;
      unsigned DD = D - DR;
      const float *l = a;
      const float *r = b;
      const float *e_l = l + DD;
      const float *e_r = r + DD;
      float unpack[8] __attribute__ ((aligned (32))) = {0};

      sum = _mm256_load_ps(unpack);
      if(DR){ AVX_DOT_FMA(e_l, e_r, sum, l0); }
      for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
        AVX_DOT_FMA(l,     r,     sum, l0);
        AVX_DOT_FMA(l + 8, r + 8, sum, l0);
      }
      _mm256_store_ps(unpack, sum);
      result = unpack[0] + unpack[1] + unpack[2] + unpack[3]
             + unpack[4] + unpack[5] + unpack[6] + unpack[7];

#else
#ifdef __SSE2__
      #define SSE_DOT_FMA(addr1, addr2, dest, tmp1) \
          tmp1 = _mm_load_ps(addr1); \
          dest = _mm_fmadd_ps(tmp1, _mm_load_ps(addr2), dest);

      __m128 sum;
      __m128 l0;
      unsigned D = (size + 3) & ~3U;
      unsigned DR = D % 16;
      unsigned DD = D - DR;
      const float *l = a;
      const float *r = b;
      const float *e_l = l + DD;
      const float *e_r = r + DD;
      float unpack[4] __attribute__ ((aligned (16))) = {0};

      sum = _mm_load_ps(unpack);
      switch (DR) {
          case 12: SSE_DOT_FMA(e_l+8, e_r+8, sum, l0);
          case 8:  SSE_DOT_FMA(e_l+4, e_r+4, sum, l0);
          case 4:  SSE_DOT_FMA(e_l, e_r, sum, l0);
        default: break;
      }
      for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
          SSE_DOT_FMA(l,     r,     sum, l0);
          SSE_DOT_FMA(l + 4, r + 4, sum, l0);
          SSE_DOT_FMA(l + 8, r + 8, sum, l0);
          SSE_DOT_FMA(l + 12,r + 12,sum, l0);
      }
      _mm_store_ps(unpack, sum);
      result += unpack[0] + unpack[1] + unpack[2] + unpack[3];
#else
      float dot0, dot1, dot2, dot3;
      const float* last = a + size;
      const float* unroll_group = last - 3;
      while (a < unroll_group) {
          dot0 = a[0] * b[0];
          dot1 = a[1] * b[1];
          dot2 = a[2] * b[2];
          dot3 = a[3] * b[3];
          result += dot0 + dot1 + dot2 + dot3;
          a += 4; b += 4;
      }
      while (a < last) { result += *a++ * *b++; }
#endif
#endif
#endif
      return result;
    }
  };

  class DistanceFastL2 : public DistanceInnerProduct{
   public:
    float norm(const float* a, unsigned size) const{
      float result = 0;
#ifdef __GNUC__
#ifdef __AVX__
      // ── AVX2 + FMA norm (a² sum) ──
      #define AVX_L2NORM_FMA(addr, dest, tmp1) \
          tmp1 = _mm256_load_ps(addr);        \
          dest = _mm256_fmadd_ps(tmp1, tmp1, dest);

      __m256 sum;
      __m256 l0;
      unsigned D = (size + 7) & ~7U;
      unsigned DR = D % 16;
      unsigned DD = D - DR;
      const float *l = a;
      const float *e_l = l + DD;
      float unpack[8] __attribute__ ((aligned (32))) = {0};

      sum = _mm256_load_ps(unpack);
      if(DR){ AVX_L2NORM_FMA(e_l, sum, l0); }
      for (unsigned i = 0; i < DD; i += 16, l += 16) {
        AVX_L2NORM_FMA(l,     sum, l0);
        AVX_L2NORM_FMA(l + 8, sum, l0);
      }
      _mm256_store_ps(unpack, sum);
      result = unpack[0] + unpack[1] + unpack[2] + unpack[3]
             + unpack[4] + unpack[5] + unpack[6] + unpack[7];
#else
#ifdef __SSE2__
      #define SSE_L2NORM_FMA(addr, dest, tmp1) \
          tmp1 = _mm_load_ps(addr);           \
          dest = _mm_fmadd_ps(tmp1, tmp1, dest);

      __m128 sum;
      __m128 l0;
      unsigned D = (size + 3) & ~3U;
      unsigned DR = D % 16;
      unsigned DD = D - DR;
      const float *l = a;
      const float *e_l = l + DD;
      float unpack[4] __attribute__ ((aligned (16))) = {0};

      sum = _mm_load_ps(unpack);
      switch (DR) {
          case 12: SSE_L2NORM_FMA(e_l+8, sum, l0);
          case 8:  SSE_L2NORM_FMA(e_l+4, sum, l0);
          case 4:  SSE_L2NORM_FMA(e_l, sum, l0);
        default: break;
      }
      for (unsigned i = 0; i < DD; i += 16, l += 16) {
          SSE_L2NORM_FMA(l,     sum, l0);
          SSE_L2NORM_FMA(l + 4, sum, l0);
          SSE_L2NORM_FMA(l + 8, sum, l0);
          SSE_L2NORM_FMA(l + 12,sum, l0);
      }
      _mm_store_ps(unpack, sum);
      result += unpack[0] + unpack[1] + unpack[2] + unpack[3];
#else
      // scalar fallback
      float dot0, dot1, dot2, dot3;
      const float* last = a + size;
      const float* unroll_group = last - 3;
      while (a < unroll_group) {
          dot0 = a[0] * a[0];
          dot1 = a[1] * a[1];
          dot2 = a[2] * a[2];
          dot3 = a[3] * a[3];
          result += dot0 + dot1 + dot2 + dot3;
          a += 4;
      }
      while (a < last) { result += (*a) * (*a); a++; }
#endif
#endif
#endif
      return result;
    }
    using DistanceInnerProduct::compare;
    float compare(const float* a, const float* b, float norm, unsigned size) const {
      float result = -2 * DistanceInnerProduct::compare(a, b, size);
      result += norm;
      return result;
    }
  };
}

#endif //EFANNA2E_DISTANCE_H
