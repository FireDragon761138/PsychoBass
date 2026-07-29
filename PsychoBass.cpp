// PsychoBass.cpp
// Psychoacoustic bass enhancer (VST 2.4, 64-bit) for Equalizer APO.
//
// Idea: each channel is split into low + high bands with a Linkwitz-Riley
// 4th-order crossover (phase-matched bands that sum to a flat allpass). The
// low band is summed to mono, split AGAIN into two sub-bands, and each
// sub-band is separately envelope-normalized and fed through a Chebyshev
// shaper (T2+T3+T4) that generates 2f/3f/4f at fixed relative levels
// regardless of input level. The ear reconstructs the "missing fundamental"
// from those harmonics, so you perceive deep bass the speakers never actually
// reproduce. Mix crossfades only the low band between real bass and the
// band-limited harmonics (out = high + (1-mix)*low + mix*harm), so the
// wet/dry sweep is free of crossover-region phase cancellation.
//
// Why two sub-bands: a memoryless nonlinearity fed two tones at once emits
// sum/difference products belonging to neither (55+70 Hz -> 125 Hz). Splitting
// first means each shaper mostly sees one note, which is what separates a
// serious virtual-bass implementation from a naive one. Each sub-band also
// gets its own harmonic ceiling at 4x ITS OWN top edge, instead of one shared
// ceiling sized for the main crossover -- deep notes no longer spray partials
// an octave higher than they can actually produce.
//
// Params:  0 Strength   level of the added harmonics   (0..1)
//          1 Crossover  split frequency, 40..400 Hz    (0..1, log)
//          2 Mix        dry (original) .. wet           (0..1)
//
// Build: see build_mingw.bat.  License: BSD-2-Clause.

#include "vst2_min.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <immintrin.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX        // keep windows.h from clobbering std::min/std::max (MSVC)
#endif
#include <windows.h>
#include <commctrl.h>   // trackbar (msctls_trackbar32)
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// The kernel MUST be inlined into each dispatch wrapper: a target region only
// governs code actually compiled inside it, so an out-of-line template
// instantiation would be emitted once at baseline and both wrappers would jump
// to the same SSE2 body. That failure is silent -- the plugin works, it just
// never runs the AVX2 path -- so build_mingw.bat greps the DLL for FMA.
#if defined(__GNUC__)
#define PB_INLINE inline __attribute__((always_inline))
#else
#define PB_INLINE __forceinline
#endif

static const VstInt32 kNumParams = 3;
enum { P_STRENGTH = 0, P_CROSSOVER = 1, P_MIX = 2 };

// Crossover knob maps 0..1 -> 40..400 Hz logarithmically.
static const double kFreqMin = 40.0;
static const double kFreqMax = 400.0;
static inline double crossoverHz(float p) {
    return kFreqMin * std::pow(kFreqMax / kFreqMin, (double)p);
}

// ------------------------------------------------------------ SIMD types ----
// ONE kernel serves both instruction sets. These are plain arithmetic vector
// types, not hand-written intrinsics: on baseline x86-64 the compiler lowers a
// vec4 op to two SSE2 instructions, and inside the AVX2 dispatch region below
// it lowers the SAME source to one AVX instruction with the multiply-adds
// contracted to FMA. No second kernel, no divergent code paths to keep in sync.
#if defined(__GNUC__)
typedef double vec2 __attribute__((vector_size(16)));
typedef double vec4 __attribute__((vector_size(32)));
static inline vec2 mk2(double a, double b) { return vec2{ a, b }; }
static inline vec4 mk4(double a, double b, double c, double d) { return vec4{ a, b, c, d }; }
static inline vec2 zero2() { return vec2{ 0.0, 0.0 }; }
static inline vec4 zero4() { return vec4{ 0.0, 0.0, 0.0, 0.0 }; }
// Element-wise min/max/abs/select. Operand order is chosen to match what
// std::min/std::max/std::fabs return for the SAME arguments, including at
// equality, so the vector form is bit-identical to the scalar form it
// replaces: _mm_min_pd(a,b) is (a<b)?a:b, and std::min(k,x) is (x<k)?x:k,
// hence min(x,k) here. There are no NaNs on this path (env is clamped above
// 1e-6), so the NaN-propagation difference between the two cannot be reached.
static PB_INLINE vec2 vmin2(vec2 a, vec2 b) { return (vec2)_mm_min_pd((__m128d)a, (__m128d)b); }
static PB_INLINE vec2 vmax2(vec2 a, vec2 b) { return (vec2)_mm_max_pd((__m128d)a, (__m128d)b); }
static PB_INLINE vec2 vabs2(vec2 a) { return (vec2)_mm_andnot_pd(_mm_set1_pd(-0.0), (__m128d)a); }
static PB_INLINE vec2 vselgt2(vec2 a, vec2 b, vec2 t, vec2 f) {
    __m128d m = _mm_cmpgt_pd((__m128d)a, (__m128d)b);
    return (vec2)_mm_or_pd(_mm_and_pd(m, (__m128d)t), _mm_andnot_pd(m, (__m128d)f));
}
#else
// MSVC has no vector_size; express the identical algebra over SSE2 registers.
// (The MSVC build is baseline-only -- see the dispatch section.)
struct vec2 {
    __m128d v;
    vec2() {}
    vec2(__m128d x) : v(x) {}
    double operator[](int i) const { double t[2]; _mm_storeu_pd(t, v); return t[i]; }
};
static inline vec2 operator*(vec2 a, vec2 b) { return _mm_mul_pd(a.v, b.v); }
static inline vec2 operator+(vec2 a, vec2 b) { return _mm_add_pd(a.v, b.v); }
static inline vec2 operator-(vec2 a, vec2 b) { return _mm_sub_pd(a.v, b.v); }
static inline vec2 operator/(vec2 a, vec2 b) { return _mm_div_pd(a.v, b.v); }
struct vec4 {
    __m128d lo, hi;
    vec4() {}
    vec4(__m128d l, __m128d h) : lo(l), hi(h) {}
    double operator[](int i) const {
        double t[4]; _mm_storeu_pd(t, lo); _mm_storeu_pd(t + 2, hi); return t[i];
    }
};
static inline vec4 operator*(vec4 a, vec4 b) { return vec4(_mm_mul_pd(a.lo, b.lo), _mm_mul_pd(a.hi, b.hi)); }
static inline vec4 operator+(vec4 a, vec4 b) { return vec4(_mm_add_pd(a.lo, b.lo), _mm_add_pd(a.hi, b.hi)); }
static inline vec4 operator-(vec4 a, vec4 b) { return vec4(_mm_sub_pd(a.lo, b.lo), _mm_sub_pd(a.hi, b.hi)); }
static inline vec2 mk2(double a, double b) { return _mm_set_pd(b, a); }
static inline vec4 mk4(double a, double b, double c, double d) { return vec4(_mm_set_pd(b, a), _mm_set_pd(d, c)); }
static inline vec2 zero2() { return _mm_setzero_pd(); }
static inline vec4 zero4() { return vec4(_mm_setzero_pd(), _mm_setzero_pd()); }
static PB_INLINE vec2 vmin2(vec2 a, vec2 b) { return _mm_min_pd(a.v, b.v); }
static PB_INLINE vec2 vmax2(vec2 a, vec2 b) { return _mm_max_pd(a.v, b.v); }
static PB_INLINE vec2 vabs2(vec2 a) { return _mm_andnot_pd(_mm_set1_pd(-0.0), a.v); }
static PB_INLINE vec2 vselgt2(vec2 a, vec2 b, vec2 t, vec2 f) {
    __m128d m = _mm_cmpgt_pd(a.v, b.v);
    return _mm_or_pd(_mm_and_pd(m, t.v), _mm_andnot_pd(m, f.v));
}
#endif

// ---------------------------------------------------------------- Biquad ----
// RBJ cookbook biquad, transposed Direct Form II. Scalar form exists only to
// DESIGN coefficients; all audio runs through the vector form below. Doubles
// throughout for clean behaviour at 40 Hz corners.
struct Biquad {
    double b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0;

    void setLowpass(double fs, double fc, double Q) {
        fc = std::min(fc, fs * 0.49);
        double w0 = 2.0 * M_PI * fc / fs;
        double cs = std::cos(w0), sn = std::sin(w0);
        double alpha = sn / (2.0 * Q);
        normalize(1.0 + alpha, (1.0 - cs) * 0.5, 1.0 - cs, (1.0 - cs) * 0.5,
                  -2.0 * cs, 1.0 - alpha);
    }

    void setHighpass(double fs, double fc, double Q) {
        fc = std::min(fc, fs * 0.49);
        double w0 = 2.0 * M_PI * fc / fs;
        double cs = std::cos(w0), sn = std::sin(w0);
        double alpha = sn / (2.0 * Q);
        normalize(1.0 + alpha, (1.0 + cs) * 0.5, -(1.0 + cs), (1.0 + cs) * 0.5,
                  -2.0 * cs, 1.0 - alpha);
    }

private:
    void normalize(double a0, double _b0, double _b1, double _b2,
                   double _a1, double _a2) {
        b0 = _b0 / a0; b1 = _b1 / a0; b2 = _b2 / a0;
        a1 = _a1 / a0; a2 = _a2 / a0;
    }
};

// N independent DF2T biquad lanes in one update. Lanes never read each other
// (every operation is element-wise), so pairing is a pure instruction-count
// win with no signal-level consequence. Lanes may carry different
// coefficients -- only state and input have to be independent.
template <class V>
struct BiquadV {
    V b0, b1, b2, a1, a2, z1, z2;
    PB_INLINE V process(V x) {
        V y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        return y;
    }
};

static inline void pack2(BiquadV<vec2>& d, const Biquad& q0, const Biquad& q1) {
    d.b0 = mk2(q0.b0, q1.b0); d.b1 = mk2(q0.b1, q1.b1); d.b2 = mk2(q0.b2, q1.b2);
    d.a1 = mk2(q0.a1, q1.a1); d.a2 = mk2(q0.a2, q1.a2);
}
// Main crossover lanes are [lowL, lowR, highL, highR]: the LP and HP sections
// see the same input, so one 4-wide update replaces two paired ones, and the
// second cascade section consumes the first's output with no lane shuffling.
static inline void pack4(BiquadV<vec4>& d, const Biquad& lo, const Biquad& hi) {
    d.b0 = mk4(lo.b0, lo.b0, hi.b0, hi.b0); d.b1 = mk4(lo.b1, lo.b1, hi.b1, hi.b1);
    d.b2 = mk4(lo.b2, lo.b2, hi.b2, hi.b2); d.a1 = mk4(lo.a1, lo.a1, hi.a1, hi.a1);
    d.a2 = mk4(lo.a2, lo.a2, hi.a2, hi.a2);
}

// Distance at which a knob smoother is treated as having arrived. See settle().
#ifndef PB_SNAP_EPS
#define PB_SNAP_EPS 1e-12
#endif

// One-pole smoother so knob moves don't zipper.
struct Smooth {
    double v = 0, coeff = 0;
    void init(double fs, double ms, double start) {
        coeff = std::exp(-1.0 / (fs * 0.001 * ms));
        v = start;
    }
    PB_INLINE double next(double target) { return v = target + coeff * (v - target); }

    // Once the remaining distance is negligible this smoother is holding a
    // constant -- in a set-and-forget host that is every block -- so snapping
    // lets the caller hoist it and everything derived from it out of the sample
    // loop. Set PB_SNAP_EPS to 0 to disable the snap and isolate this from the
    // rest of the arithmetic when null-testing.
    //
    // 1e-12 is chosen from both ends. It is four orders of magnitude below the
    // resolution of the float32 samples this ends up in (2^-23 = 1.2e-7), so
    // the largest output change it can cause is a single float ULP where it
    // tips a rounding boundary. And the smoother reaches it 28 time constants
    // after a knob move -- 0.83 s at 30 ms -- so the hoist re-engages promptly.
    // Left alone, FTZ eventually drives the recursion to exact equality anyway,
    // but that takes ~21 s and leans on denormal flushing to do it.
    bool settle(double target) {
        if (std::fabs(v - target) > PB_SNAP_EPS) return false;
        v = target;
        return true;
    }
};

// ---------------------------------------------------------------- Plugin ----
struct PsychoBass {
    AEffect effect;
    audioMasterCallback master = nullptr;

    float  params[kNumParams];
    double fs = 44100.0;

    // Main crossover: two identical 12 dB/oct sections cascaded -> Linkwitz-
    // Riley 24 dB/oct. LP and HP outputs are phase-matched and sum to a flat
    // allpass, so the low band can be crossfaded against the harmonics without
    // cancellation at any Mix setting. Lanes: [lowL, lowR, highL, highR].
    BiquadV<vec4> xover[2];
    // Sub-split of the mono bass, LR4 at fSplit. Lanes: [bandA, bandB].
    // LR4 again so bandA + bandB reconstructs the bass with no level notch.
    BiquadV<vec2> sub[2];
    // Harmonic cleanup, lanes = [bandA, bandB]. HP strips DC + residual
    // fundamental (both lanes at fc); LP caps each band at 4x its own top edge.
    BiquadV<vec2> hpHarm[2];
    BiquadV<vec2> lpHarm[2];

    Smooth smStrength, smMix, smGain;
    double lastCrossover = -1.0;

    // Envelope followers: one per sub-band to normalize its shaper input (both
    // lanes of one vec2 -- they arrive already packed from the sub-split), one
    // on the mono bass as the loudness-match reference, one on the harmonic
    // sum as the loudness-match measurement.
    vec2 envAB;
    double envRef = 0.0, envHarm = 0.0;
    double envAtk = 0.0, envRel = 0.0;

    // Over-aligned members (vec4 wants 32) -- don't rely on the default
    // operator new's 16-byte guarantee.
    static void* operator new(size_t n) {
#if defined(_WIN32)
        void* p = _aligned_malloc(n, 32);
#else
        void* p = std::aligned_alloc(32, (n + 31) & ~size_t(31));
#endif
        return p;
    }
    static void operator delete(void* p) {
#if defined(_WIN32)
        _aligned_free(p);
#else
        std::free(p);
#endif
    }

#if defined(_WIN32)
    // Editor (native Win32) -- Equalizer APO has no generic slider fallback, so
    // the plugin must provide its own GUI.
    HWND edContainer = nullptr;
    HWND edSlider[kNumParams] = { nullptr, nullptr, nullptr };
    HWND edValue[kNumParams]  = { nullptr, nullptr, nullptr };
    void openEditor(HWND parent);
    void closeEditor();
    void refreshValue(int i);
    void resetDefaults();
#endif

    // Single source of truth for factory values, so the Defaults button and
    // the constructor cannot drift apart.
    void setDefaultParams() {
        params[P_STRENGTH]  = 0.5f;
        params[P_CROSSOVER] = 0.5f;   // ~126 Hz
        params[P_MIX]       = 0.75f;
    }

    PsychoBass() {
        setDefaultParams();
        setSampleRate(44100.0);
        // Filter and envelope state has no in-class initializer (operator new
        // hands back uninitialized storage), so clear it here rather than
        // relying on the host to send effMainsChanged before the first block.
        resetState();
    }

    void setSampleRate(double sr) {
        fs = sr;
        smStrength.init(fs, 30.0, params[P_STRENGTH]);
        smMix.init(fs, 30.0, params[P_MIX]);
        smGain.init(fs, 50.0, 1.0);
        envAtk = std::exp(-1.0 / (fs * 0.005));   // 5 ms attack
        envRel = std::exp(-1.0 / (fs * 0.120));   // 120 ms release
        lastCrossover = -1.0;      // force coeff recompute
        updateFilters(crossoverHz(params[P_CROSSOVER]), true);
    }

    void resetState() {
        for (int i = 0; i < 2; ++i) {
            xover[i].z1 = xover[i].z2 = zero4();
            sub[i].z1 = sub[i].z2 = zero2();
            hpHarm[i].z1 = hpHarm[i].z2 = zero2();
            lpHarm[i].z1 = lpHarm[i].z2 = zero2();
        }
        envAB = zero2();
        envRef = envHarm = 0.0;
        smGain.v = 1.0;
    }

    // Butterworth cascade Q's (maximally flat 24 dB/oct).
    static constexpr double Q1 = 0.54119610;
    static constexpr double Q2 = 1.30656296;
    // Linkwitz-Riley 4th order = two identical Butterworth (Q = 1/sqrt2) sections.
    static constexpr double kQLR = 0.70710678118654752;
    // Sub-split sits a half-octave below the main crossover: low enough to put
    // a kick fundamental and a bass note in different bands (the whole point),
    // high enough that band A still carries most of the energy.
    static constexpr double kSubSplit = 0.70710678118654752;
    // Band B's notes sit nearer the crossover, so the transducer already
    // reproduces more of them and they need proportionally less synthesis.
    static constexpr double kWeightB = 0.6;
    // Reciprocal of the gate threshold (~-70 dBFS). A linear ramp to unity
    // costs a multiply where env/(env+k) cost a division; both are fully
    // faded out well below anything audible, so the curve shape is free.
    static constexpr double kInvGate = 1.0 / 3e-4;

    void updateFilters(double fc, bool force) {
        if (!force && std::fabs(fc - lastCrossover) < 0.5) return;
        lastCrossover = fc;

        const double fSplit = fc * kSubSplit;
        const double ceiling = std::min(6000.0, fs * 0.45);

        Biquad lo, hi;
        lo.setLowpass(fs, fc, kQLR);
        hi.setHighpass(fs, fc, kQLR);
        pack4(xover[0], lo, hi);
        pack4(xover[1], lo, hi);

        lo.setLowpass(fs, fSplit, kQLR);
        hi.setHighpass(fs, fSplit, kQLR);
        pack2(sub[0], lo, hi);
        pack2(sub[1], lo, hi);

        Biquad h1, h2;
        h1.setHighpass(fs, fc, Q1);
        h2.setHighpass(fs, fc, Q2);
        pack2(hpHarm[0], h1, h1);
        pack2(hpHarm[1], h2, h2);

        // Each band's harmonics stop two octaves above ITS OWN top edge --
        // band A's 4th harmonic cannot exceed 4*fSplit, so giving it the old
        // shared 4*fc ceiling just let intermodulation and crossover-shoulder
        // content through an octave above anything it could legitimately make.
        Biquad la1, la2, lb1, lb2;
        la1.setLowpass(fs, std::min(fSplit * 4.0, ceiling), Q1);
        la2.setLowpass(fs, std::min(fSplit * 4.0, ceiling), Q2);
        lb1.setLowpass(fs, std::min(fc * 4.0, ceiling), Q1);
        lb2.setLowpass(fs, std::min(fc * 4.0, ceiling), Q2);
        pack2(lpHarm[0], la1, lb1);
        pack2(lpHarm[1], la2, lb2);
    }

    // Harmonic generator: Chebyshev polynomials on a unit-normalized input
    // give exact per-harmonic levels for a sine (T2 -> 2f, T3 -> 3f,
    // T4 -> 4f), weighted -6 dB per step so the stack reads as bass, not buzz.
    // Both sub-bands at once: they arrive packed from the sub-split and stay
    // packed through the shaper and the cleanup filters, so the pairing costs
    // no shuffling anywhere along the path.
    static PB_INLINE vec2 cheby2(vec2 x) {
        const vec2 one = mk2(1.0, 1.0);
        vec2 x2 = x * x;
        vec2 t2 = mk2(2.0, 2.0) * x2 - one;
        vec2 t3 = x * (mk2(4.0, 4.0) * x2 - mk2(3.0, 3.0));
        vec2 t4 = mk2(8.0, 8.0) * x2 * (x2 - one) + one;
        return t2 + mk2(0.5, 0.5) * t3 + mk2(0.25, 0.25) * t4;
    }

    // Envelope update for the two sub-bands. The select is a compare plus an
    // and/andnot/or on baseline SSE2 -- three ops where the scalar form used
    // one cmov -- but it covers BOTH bands, so the pair still comes out ahead
    // of two scalar updates, and the inputs are already in a register pair.
    PB_INLINE vec2 advanceEnv2(vec2 x, vec2 atk, vec2 rel) {
        vec2 a = vabs2(x);
        vec2 c = vselgt2(a, envAB, atk, rel);
        envAB = a + c * (envAB - a);
        return envAB;
    }

    // Scalar form, still used for the two loudness-match envelopes: those sit
    // at opposite ends of the chain (one on the input bass, one on the finished
    // harmonics) and have nothing to pair with.
    PB_INLINE double advanceEnv(double x, double& env) const {
        double a = std::fabs(x);
        env = a + (a > env ? envAtk : envRel) * (env - a);
        return env;
    }

    // Chebyshev separation is exact only at unit amplitude, which is why the
    // input is normalized and the envelope re-applied after -- off unit
    // amplitude the polynomials leak DC and fundamental, and the cleanup
    // filters have to catch it downstream. Takes a precomputed reciprocal so
    // both bands' divisions can be issued as one paired instruction.
    static PB_INLINE vec2 shape2(vec2 x, vec2 env, vec2 rcpEnv, vec2 invGate) {
        const vec2 one = mk2(1.0, 1.0);
        vec2 xn = vmin2(x * rcpEnv, one);
        xn = vmax2(xn, mk2(-1.0, -1.0));
        vec2 gate = vmin2(env * invGate, one);        // fade out below ~-70 dBFS
        return cheby2(xn) * env * gate;
    }

    // ------------------------------------------------------------ kernel ----
    // ONE kernel. Compiled twice -- once at baseline (SSE2, vec4 -> two
    // 128-bit ops) and once inside the AVX2 target region (vec4 -> one 256-bit
    // op, multiply-adds contracted to FMA) -- and selected per CPU at load.
    template <typename T>
    PB_INLINE void kernel(T** in, T** out, VstInt32 n) {
        // Flush subnormals to zero: after the input goes silent the biquad
        // states decay into denormal range and stall the FPU's fast path.
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

        // Recompute crossover coeffs once per block from the current knob.
        updateFilters(crossoverHz(params[P_CROSSOVER]), false);

        const T* inL = in[0];
        const T* inR = in[1];
        T* outL = out[0];
        T* outR = out[1];

        // Loudness-match target, evaluated once per block. It feeds a 50 ms
        // smoother, so a block of staleness (~11 ms at 512/48k) is far below
        // what that smoother resolves -- and it keeps a division off the
        // per-sample serial dependency chain.
        const double gainTarget = std::min(envRef / std::max(envHarm, 1e-9), 4.0);

        // Knob smoothing costs two serial one-poles per sample and exists only
        // to cover a knob actually in motion. Once both have closed on their
        // targets they are constants, so hoist them and the dry coefficient
        // derived from them; the branch below then runs one way for the whole
        // block and predicts perfectly. In a set-and-forget host this is every
        // block, which is why it is worth a branch rather than a second loop.
        const double strTarget = params[P_STRENGTH];
        const double mixTarget = params[P_MIX];
        const bool strStatic = smStrength.settle(strTarget);
        const bool mixStatic = smMix.settle(mixTarget);
        const bool ctlStatic = strStatic && mixStatic;

        double strength = smStrength.v;
        double mix      = smMix.v;
        double dryGain  = 1.0 - mix;

        const vec2 vAtk  = mk2(envAtk, envAtk);
        const vec2 vRel  = mk2(envRel, envRel);
        const vec2 vGate = mk2(kInvGate, kInvGate);
        const vec2 vTiny = mk2(1e-6, 1e-6);
        const vec2 vOne  = mk2(1.0, 1.0);

        for (VstInt32 i = 0; i < n; ++i) {
            if (!ctlStatic) {
                strength = smStrength.next(strTarget);
                mix      = smMix.next(mixTarget);
                dryGain  = 1.0 - mix;
            }

            double L = (double)inL[i];
            double R = (double)inR[i];

            // --- LR4 band split; one 4-wide cascade yields both bands for
            //     both channels, lanes already in the order we consume them ---
            vec4 v = xover[1].process(xover[0].process(mk4(L, R, L, R)));
            double lowL = v[0], lowR = v[1], highL = v[2], highR = v[3];

            // --- harmonic path (mono; equals the split of the mono sum by
            //     linearity, and bass is near-mono anyway) ---
            double bass = 0.5 * (lowL + lowR);

            advanceEnv(bass, envRef);

            // Two sub-bands so simultaneous notes land in separate shapers and
            // cannot intermodulate with each other.
            vec2 b = sub[1].process(sub[0].process(mk2(bass, bass)));
            vec2 e = advanceEnv2(b, vAtk, vRel);

            // Both bands' normalizations issue as ONE divide. Division is the
            // most expensive thing in this loop and it sits on the serial
            // path, so halving the count matters more than the arithmetic.
            vec2 rcp = vOne / vmax2(e, vTiny);
            vec2 hv = shape2(b, e, rcp, vGate);

            hv = hpHarm[1].process(hpHarm[0].process(hv));   // strip DC + fundamental
            hv = lpHarm[1].process(lpHarm[0].process(hv));   // per-band ceiling
            double h = hv[0] + kWeightB * hv[1];

            // --- loudness match: hold the synthesized level to the level of
            //     the bass being removed, so Strength sets a ratio instead of
            //     a raw gain that drifts with how much of the harmonic series
            //     survived the filters. Clamped so a note whose partials all
            //     fall outside the passband can't run away. ---
            advanceEnv(h, envHarm);
            h *= smGain.next(gainTarget) * strength * 0.8;

            // --- phase-coherent wet/dry: only the low band crossfades (real
            // bass vs harmonics), so no Mix setting can cancel against the
            // high band. ---
            outL[i] = (T)(highL + dryGain * lowL + mix * h);
            outR[i] = (T)(highR + dryGain * lowR + mix * h);
        }
    }
};

// -------------------------------------------------------------- dispatch ----
// The same kernel, emitted twice, chosen once at load. One DLL runs optimally
// on a 2006 Core 2 and a 2024 Ryzen; there is no build variant to install and
// nothing to crash on an old CPU.
static void runF_base(PsychoBass* p, float**  in, float**  out, VstInt32 n) { p->kernel<float>(in, out, n); }
static void runD_base(PsychoBass* p, double** in, double** out, VstInt32 n) { p->kernel<double>(in, out, n); }

// PB_BASELINE_ONLY drops the second emission (measurement / size-constrained
// builds). The plugin stays correct, just never faster than SSE2.
#if defined(__GNUC__) && !defined(PB_BASELINE_ONLY)
#define PB_HAVE_AVX2 1
#pragma GCC push_options
#pragma GCC target("avx2,fma")
static void runF_avx2(PsychoBass* p, float**  in, float**  out, VstInt32 n) { p->kernel<float>(in, out, n); }
static void runD_avx2(PsychoBass* p, double** in, double** out, VstInt32 n) { p->kernel<double>(in, out, n); }
#pragma GCC pop_options
#endif

typedef void (*ProcFloatFn)(PsychoBass*, float**, float**, VstInt32);
typedef void (*ProcDoubleFn)(PsychoBass*, double**, double**, VstInt32);

static bool hostHasAvx2() {
#if defined(PB_HAVE_AVX2)
    __builtin_cpu_init();
    // GCC's detector already gates AVX on OSXSAVE + XCR0, so a CPU with AVX2
    // but an OS that doesn't save YMM state correctly reports false.
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

#if defined(PB_HAVE_AVX2)
static ProcFloatFn  g_procFloat  = hostHasAvx2() ? runF_avx2 : runF_base;
static ProcDoubleFn g_procDouble = hostHasAvx2() ? runD_avx2 : runD_base;
#else
static ProcFloatFn  g_procFloat  = runF_base;
static ProcDoubleFn g_procDouble = runD_base;
#endif

// Reported in the editor so the active path is visible, not assumed.
static const char* activePathName() {
#if defined(PB_HAVE_AVX2)
    return (g_procFloat == runF_avx2) ? "AVX2+FMA" : "SSE2";
#else
    return "SSE2";
#endif
}

// ------------------------------------------------------------- editor GUI ----
#if defined(_WIN32)
// Fixed editor size the host reads via effEditGetRect (top,left,bottom,right).
static VstRect g_edRect = { 0, 0, 200, 448 };
enum { kResetId = 200 };

static HINSTANCE dllInstance() {
    HMODULE h = nullptr;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCSTR)&g_edRect, &h);
    return (HINSTANCE)h;
}

void PsychoBass::refreshValue(int i) {
    if (!edValue[i]) return;
    char buf[32];
    double v = params[i];
    if (i == P_CROSSOVER) std::snprintf(buf, sizeof buf, "%.0f Hz", crossoverHz((float)v));
    else                  std::snprintf(buf, sizeof buf, "%.0f %%", v * 100.0);
    SetWindowTextA(edValue[i], buf);
}

void PsychoBass::resetDefaults() {
    setDefaultParams();
    for (int i = 0; i < kNumParams; ++i) {
        if (edSlider[i]) SendMessageA(edSlider[i], TBM_SETPOS, TRUE, (LPARAM)(params[i] * 1000.0f));
        refreshValue(i);
    }
}

static LRESULT CALLBACK EditorWndProc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_CREATE) {
        CREATESTRUCTA* cs = (CREATESTRUCTA*)lp;
        SetWindowLongPtrA(h, GWLP_USERDATA, (LONG_PTR)cs->lpCreateParams);
        return 0;
    }
    PsychoBass* p = (PsychoBass*)GetWindowLongPtrA(h, GWLP_USERDATA);
    if (msg == WM_HSCROLL && p) {
        HWND tb = (HWND)lp;
        for (int i = 0; i < kNumParams; ++i) {
            if (tb == p->edSlider[i]) {
                int pos = (int)SendMessageA(tb, TBM_GETPOS, 0, 0);
                p->params[i] = (float)(pos / 1000.0);
                p->refreshValue(i);
                break;
            }
        }
        return 0;
    }
    if (msg == WM_COMMAND && p && LOWORD(wp) == kResetId) {
        p->resetDefaults();
        return 0;
    }
    if (msg == WM_CTLCOLORSTATIC) return (LRESULT)GetSysColorBrush(COLOR_BTNFACE);
    return DefWindowProcA(h, msg, wp, lp);
}

void PsychoBass::openEditor(HWND parent) {
    if (edContainer) return;
    HINSTANCE inst = dllInstance();

    INITCOMMONCONTROLSEX icc = { sizeof(icc), ICC_BAR_CLASSES };
    InitCommonControlsEx(&icc);

    static const char* kClass = "PsychoBassEditorWnd";
    WNDCLASSEXA wc;
    if (!GetClassInfoExA(inst, kClass, &wc)) {
        ZeroMemory(&wc, sizeof wc);
        wc.cbSize        = sizeof wc;
        wc.lpfnWndProc   = EditorWndProc;
        wc.hInstance     = inst;
        wc.lpszClassName = kClass;
        wc.hCursor       = LoadCursor(nullptr, IDC_ARROW);
        wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);
        RegisterClassExA(&wc);
    }

    edContainer = CreateWindowExA(0, kClass, "", WS_CHILD | WS_VISIBLE,
                                  0, 0, g_edRect.right, g_edRect.bottom,
                                  parent, nullptr, inst, this);
    if (!edContainer) return;

    const char* names[kNumParams] = { "Strength", "Crossover", "Wet / Dry" };
    for (int i = 0; i < kNumParams; ++i) {
        int y = 22 + i * 56;
        CreateWindowExA(0, "STATIC", names[i], WS_CHILD | WS_VISIBLE,
                        16, y, 96, 20, edContainer, nullptr, inst, nullptr);
        HWND tb = CreateWindowExA(0, TRACKBAR_CLASSA, "",
                        WS_CHILD | WS_VISIBLE | TBS_HORZ | TBS_NOTICKS,
                        116, y - 2, 250, 28, edContainer,
                        (HMENU)(intptr_t)(100 + i), inst, nullptr);
        SendMessageA(tb, TBM_SETRANGE, TRUE, MAKELONG(0, 1000));
        SendMessageA(tb, TBM_SETPOS, TRUE, (LPARAM)(params[i] * 1000.0f));
        edSlider[i] = tb;
        edValue[i] = CreateWindowExA(0, "STATIC", "", WS_CHILD | WS_VISIBLE,
                        374, y, 68, 20, edContainer, nullptr, inst, nullptr);
        refreshValue(i);
    }

    char foot[64];
    std::snprintf(foot, sizeof foot, "PsychoBass  -  %s", activePathName());
    CreateWindowExA(0, "STATIC", foot, WS_CHILD | WS_VISIBLE,
                    16, 172, 240, 18, edContainer, nullptr, inst, nullptr);
    CreateWindowExA(0, "BUTTON", "Defaults", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                    374, 168, 68, 24, edContainer,
                    (HMENU)(intptr_t)kResetId, inst, nullptr);
}

void PsychoBass::closeEditor() {
    if (edContainer) { DestroyWindow(edContainer); edContainer = nullptr; }
    for (int i = 0; i < kNumParams; ++i) { edSlider[i] = nullptr; edValue[i] = nullptr; }
}
#endif // _WIN32

// ---------------------------------------------------- host entry helpers ----
static void setParameter(AEffect* e, VstInt32 index, float value) {
    PsychoBass* p = (PsychoBass*)e->object;
    if (index >= 0 && index < kNumParams) p->params[index] = value;
}
static float getParameter(AEffect* e, VstInt32 index) {
    PsychoBass* p = (PsychoBass*)e->object;
    return (index >= 0 && index < kNumParams) ? p->params[index] : 0.0f;
}
static void processReplacing(AEffect* e, float** in, float** out, VstInt32 n) {
    g_procFloat((PsychoBass*)e->object, in, out, n);
}
static void processDoubleReplacing(AEffect* e, double** in, double** out, VstInt32 n) {
    g_procDouble((PsychoBass*)e->object, in, out, n);
}

// Bounded copy that writes ONLY the needed bytes + terminator (never pads to
// cap). VST2 guarantees small fixed buffers (e.g. kVstMaxParamStrLen == 8 for
// param strings); strncpy's zero-fill would overrun them.
static void copyStr(void* dst, const char* s, size_t cap) {
    if (!dst || cap == 0) return;
    char* d = (char*)dst;
    size_t i = 0;
    for (; s[i] && i + 1 < cap; ++i) d[i] = s[i];
    d[i] = 0;
}

// VST2 max string lengths (bytes) the host allocates.
enum {
    kMaxParamStr   = 8,    // kVstMaxParamStrLen
    kMaxProgName   = 24,   // kVstMaxProgNameLen
    kMaxEffectName = 32,   // kVstMaxEffectNameLen
    kMaxVendorStr  = 64,   // kVstMaxVendorStrLen
    kMaxProductStr = 64    // kVstMaxProductStrLen
};

static VstIntPtr dispatcher(AEffect* e, VstInt32 opcode, VstInt32 index,
                            VstIntPtr value, void* ptr, float opt) {
    PsychoBass* p = (PsychoBass*)e->object;
    switch (opcode) {
    case effOpen:  return 0;
    case effClose:
        if (p) {
#if defined(_WIN32)
            p->closeEditor();
#endif
            delete p; e->object = nullptr;
        }
        return 0;

    case effSetSampleRate: p->setSampleRate((double)opt); return 0;
    case effSetBlockSize:  return 0;
    case effMainsChanged:  if (value) p->resetState(); return 0;

#if defined(_WIN32)
    // Editor. Equalizer APO calls effEditGetRect and dereferences the returned
    // pointer WITHOUT null-checking, so we must always hand back a valid rect.
    case effEditGetRect:
        if (ptr) *(VstRect**)ptr = &g_edRect;
        return 1;
    case effEditOpen:
        if (p) p->openEditor((HWND)ptr);
        return 1;
    case effEditClose:
        if (p) p->closeEditor();
        return 1;
    case effEditIdle:
        return 0;
#endif

    case effGetParamName:
    case effGetParamLabel:
    case effGetParamDisplay: {
        char buf[32] = {0};
        double v = (index >= 0 && index < kNumParams) ? p->params[index] : 0.0;
        if (index < 0 || index >= kNumParams) { copyStr(ptr, "", kMaxParamStr); return 0; }
        if (opcode == effGetParamName) {
            // Must fit kVstMaxParamStrLen (8 bytes -> 7 chars).
            const char* names[] = { "Amount", "Xover", "Mix" };
            copyStr(ptr, names[index], kMaxParamStr);
        } else if (opcode == effGetParamLabel) {
            const char* labels[] = { "%", "Hz", "%" };
            copyStr(ptr, labels[index], kMaxParamStr);
        } else { // display
            if (index == P_CROSSOVER) std::snprintf(buf, sizeof buf, "%.0f", crossoverHz((float)v));
            else                      std::snprintf(buf, sizeof buf, "%.0f", v * 100.0);
            copyStr(ptr, buf, kMaxParamStr);
        }
        return 0;
    }

    case effCanBeAutomated: return 1;

    case effGetEffectName:   copyStr(ptr, "PsychoBass", kMaxEffectName); return 1;
    case effGetProductString:copyStr(ptr, "PsychoBass", kMaxProductStr); return 1;
    case effGetVendorString: copyStr(ptr, "daede", kMaxVendorStr);       return 1;
    case effGetVendorVersion: return 1100;
    case effGetPlugCategory:  return kPlugCategEffect;
    case effGetVstVersion:    return 2400;

    case effCanDo:
        // We only do audio; no events/programs beyond the basics.
        return 0;

    // Programs (single default program).
    case effGetProgramName: copyStr(ptr, "Default", kMaxProgName); return 0;
    case effSetProgramName: return 0;
    case effGetProgram:     return 0;
    case effSetProgram:     return 0;

    default: return 0;
    }
}

#if defined(_WIN32)
#define VST_EXPORT extern "C" __declspec(dllexport)
#else
#define VST_EXPORT extern "C" __attribute__((visibility("default")))
#endif

VST_EXPORT AEffect* VSTPluginMain(audioMasterCallback audioMaster) {
    PsychoBass* p = new PsychoBass();
    AEffect* e = &p->effect;
    std::memset(e, 0, sizeof(AEffect));

    e->magic      = kEffectMagic;
    e->dispatcher = dispatcher;
    e->setParameter = setParameter;
    e->getParameter = getParameter;
    e->processReplacing       = processReplacing;
    e->processDoubleReplacing = processDoubleReplacing;

    e->numPrograms = 1;
    e->numParams   = kNumParams;
    e->numInputs   = 2;
    e->numOutputs  = 2;
    e->flags       = effFlagsCanReplacing | effFlagsCanDoubleReplacing
                   | effFlagsHasEditor;
    e->uniqueID    = CCONST('P', 'b', 'a', 's');  // 'Pbas'
    e->version     = 1100;
    e->object      = p;

    p->master = audioMaster;
    return e;
}

// Some hosts still look for the legacy "main" entry point.
#if defined(_WIN32)
VST_EXPORT AEffect* MAIN(audioMasterCallback audioMaster) {
    return VSTPluginMain(audioMaster);
}
#endif
