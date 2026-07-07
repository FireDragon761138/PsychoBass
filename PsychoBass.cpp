// PsychoBass.cpp
// Psychoacoustic bass enhancer (VST 2.4, 64-bit) for Equalizer APO.
//
// Idea: each channel is split into low + high bands with a Linkwitz-Riley
// 4th-order crossover (phase-matched bands that sum to a flat allpass). The
// low band is envelope-normalized and fed through a Chebyshev shaper
// (T2+T3+T4) that generates 2f/3f/4f at fixed relative levels regardless of
// input level. The ear reconstructs the "missing fundamental" from those
// harmonics, so you perceive deep bass that the speakers never actually
// reproduce. Mix crossfades only the low band between real bass and the
// band-limited harmonics (out = high + (1-mix)*low + mix*harm), so the
// wet/dry sweep is free of crossover-region phase cancellation.
//
// Params:  0 Strength   level of the added harmonics   (0..1)
//          1 Crossover  split frequency, 40..400 Hz    (0..1, log)
//          2 Mix        dry (original) .. wet           (0..1)
//
// Build: see build.bat.  License: BSD-2-Clause.

#include "vst2_min.h"
#include <cmath>
#include <cstring>
#include <cstdio>
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

static const VstInt32 kNumParams = 3;
enum { P_STRENGTH = 0, P_CROSSOVER = 1, P_MIX = 2 };

// Crossover knob maps 0..1 -> 40..400 Hz logarithmically.
static const double kFreqMin = 40.0;
static const double kFreqMax = 400.0;
static inline double crossoverHz(float p) {
    return kFreqMin * std::pow(kFreqMax / kFreqMin, (double)p);
}

// ---------------------------------------------------------------- Biquad ----
// RBJ cookbook biquad, transposed Direct Form II. Coeffs are double for clean
// low-frequency behaviour; sample state templated so we can run float or double
// process paths through the same class.
struct Biquad {
    double b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0;
    double z1 = 0, z2 = 0;

    void reset() { z1 = z2 = 0; }

    void setLowpass(double fs, double fc, double Q) {
        fc = std::min(fc, fs * 0.49);
        double w0 = 2.0 * M_PI * fc / fs;
        double cs = std::cos(w0), sn = std::sin(w0);
        double alpha = sn / (2.0 * Q);
        double a0 =  1.0 + alpha;
        double _b0 = (1.0 - cs) * 0.5;
        double _b1 =  1.0 - cs;
        double _b2 = (1.0 - cs) * 0.5;
        double _a1 = -2.0 * cs;
        double _a2 =  1.0 - alpha;
        normalize(a0, _b0, _b1, _b2, _a1, _a2);
    }

    void setHighpass(double fs, double fc, double Q) {
        fc = std::min(fc, fs * 0.49);
        double w0 = 2.0 * M_PI * fc / fs;
        double cs = std::cos(w0), sn = std::sin(w0);
        double alpha = sn / (2.0 * Q);
        double a0 =  1.0 + alpha;
        double _b0 = (1.0 + cs) * 0.5;
        double _b1 = -(1.0 + cs);
        double _b2 = (1.0 + cs) * 0.5;
        double _a1 = -2.0 * cs;
        double _a2 =  1.0 - alpha;
        normalize(a0, _b0, _b1, _b2, _a1, _a2);
    }

    inline double process(double x) {
        double y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        return y;
    }

private:
    void normalize(double a0, double _b0, double _b1, double _b2,
                   double _a1, double _a2) {
        b0 = _b0 / a0; b1 = _b1 / a0; b2 = _b2 / a0;
        a1 = _a1 / a0; a2 = _a2 / a0;
    }
};

// One-pole smoother so knob moves don't zipper.
struct Smooth {
    double v = 0, coeff = 0;
    void init(double fs, double ms, double start) {
        coeff = std::exp(-1.0 / (fs * 0.001 * ms));
        v = start;
    }
    inline double next(double target) { return v = target + coeff * (v - target); }
};

// ---------------------------------------------------------------- Plugin ----
struct PsychoBass {
    AEffect effect;
    audioMasterCallback master = nullptr;

    float  params[kNumParams];
    double fs = 44100.0;

    // Main crossover: two identical 12 dB/oct sections cascaded per band ->
    // Linkwitz-Riley 24 dB/oct. LP and HP outputs are phase-matched and sum
    // to a flat-magnitude allpass, so the low band can be crossfaded against
    // the harmonics without cancellation at any Mix setting.
    Biquad lpMainL[2], lpMainR[2]; // isolate the low band (per channel)
    Biquad hpMainL[2], hpMainR[2]; // the untouched high band (per channel)
    Biquad hpHarm[2];           // remove DC + residual fundamental from harmonics
    Biquad lpHarm[2];           // tame harmonic brightness (24 dB/oct)

    Smooth smStrength, smMix;
    double lastCrossover = -1.0;

    // Bass-band envelope follower: normalizes the shaper input so the
    // harmonic recipe is level-independent.
    double env = 0.0;
    double envAtk = 0.0, envRel = 0.0;

#if defined(_WIN32)
    // Editor (native Win32) -- Equalizer APO has no generic slider fallback, so
    // the plugin must provide its own GUI.
    HWND edContainer = nullptr;
    HWND edSlider[kNumParams] = { nullptr, nullptr, nullptr };
    HWND edValue[kNumParams]  = { nullptr, nullptr, nullptr };
    void openEditor(HWND parent);
    void closeEditor();
    void refreshValue(int i);
#endif

    PsychoBass() {
        params[P_STRENGTH]  = 0.5f;
        params[P_CROSSOVER] = 0.5f;   // ~126 Hz
        params[P_MIX]       = 0.5f;
        setSampleRate(44100.0);
    }

    void setSampleRate(double sr) {
        fs = sr;
        smStrength.init(fs, 30.0, params[P_STRENGTH]);
        smMix.init(fs, 30.0, params[P_MIX]);
        envAtk = std::exp(-1.0 / (fs * 0.005));   // 5 ms attack
        envRel = std::exp(-1.0 / (fs * 0.120));   // 120 ms release
        lastCrossover = -1.0;      // force coeff recompute
        updateFilters(crossoverHz(params[P_CROSSOVER]), true);
    }

    void resetState() {
        for (int i = 0; i < 2; ++i) {
            lpMainL[i].reset(); lpMainR[i].reset();
            hpMainL[i].reset(); hpMainR[i].reset();
            hpHarm[i].reset(); lpHarm[i].reset();
        }
        env = 0.0;
    }

    // Harmonic band-limiting: Butterworth cascade Q's (maximally flat 24 dB/oct).
    static constexpr double Q1 = 0.54119610;
    static constexpr double Q2 = 1.30656296;
    // Main crossover: Linkwitz-Riley 4th order = two identical 2nd-order
    // Butterworth (Q = 1/sqrt2) sections cascaded.
    static constexpr double kQLR = 0.70710678118654752;

    void updateFilters(double fc, bool force) {
        if (!force && std::fabs(fc - lastCrossover) < 0.5) return;
        lastCrossover = fc;

        for (int i = 0; i < 2; ++i) {
            lpMainL[i].setLowpass(fs, fc, kQLR);
            lpMainR[i].setLowpass(fs, fc, kQLR);
            hpMainL[i].setHighpass(fs, fc, kQLR);
            hpMainR[i].setHighpass(fs, fc, kQLR);
        }
        hpHarm[0].setHighpass(fs, fc, Q1);
        hpHarm[1].setHighpass(fs, fc, Q2);

        // Keep harmonics musical: the illusion only needs 2f..4f, so cut hard
        // (24 dB/oct) two octaves above the crossover — content above that
        // reads as midrange distortion, not bass.
        double fHarmLP = std::min(fc * 4.0, std::min(6000.0, fs * 0.45));
        lpHarm[0].setLowpass(fs, fHarmLP, Q1);
        lpHarm[1].setLowpass(fs, fHarmLP, Q2);
    }

    // Harmonic generator: Chebyshev polynomials on a unit-normalized input
    // give exact per-harmonic levels for a sine (T2 -> 2f, T3 -> 3f,
    // T4 -> 4f), weighted -6 dB per step so the stack reads as bass, not buzz.
    static inline double cheby(double x) {
        double x2 = x * x;
        double t2 = 2.0 * x2 - 1.0;
        double t3 = x * (4.0 * x2 - 3.0);
        double t4 = 8.0 * x2 * (x2 - 1.0) + 1.0;
        return t2 + 0.5 * t3 + 0.25 * t4;
    }

    template <typename T>
    void run(T** in, T** out, VstInt32 n) {
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

        for (VstInt32 i = 0; i < n; ++i) {
            double strength = smStrength.next(params[P_STRENGTH]);
            double mix      = smMix.next(params[P_MIX]);

            double L = (double)inL[i];
            double R = (double)inR[i];

            // --- LR4 band split (low + high are phase-matched, sum flat) ---
            double lowL = lpMainL[0].process(L);  lowL = lpMainL[1].process(lowL);
            double lowR = lpMainR[0].process(R);  lowR = lpMainR[1].process(lowR);
            double highL = hpMainL[0].process(L); highL = hpMainL[1].process(highL);
            double highR = hpMainR[0].process(R); highR = hpMainR[1].process(highR);

            // --- harmonic path (mono; equals LP of the mono sum by linearity) ---
            double bass = 0.5 * (lowL + lowR);

            // Envelope-normalize into the shaper, re-apply the envelope
            // after: constant harmonic recipe at every playback level.
            double a = std::fabs(bass);
            env = a + (a > env ? envAtk : envRel) * (env - a);
            double xn = bass / std::max(env, 1e-6);
            xn = std::max(-1.0, std::min(1.0, xn));
            double gate = env / (env + 3e-4);  // fade out below ~-70 dBFS

            double h = cheby(xn) * env * gate;
            h = hpHarm[0].process(h);
            h = hpHarm[1].process(h);     // strip DC + residual fundamental
            h = lpHarm[0].process(h);     // band-limit the harmonics
            h = lpHarm[1].process(h);
            h *= strength * 0.8;          // strength sets harmonic level

            // --- phase-coherent wet/dry: only the low band crossfades (real
            // bass vs harmonics), so no Mix setting can cancel against the
            // high band. Algebraically identical to a global dry/wet blend
            // with an allpass-matched dry path. ---
            outL[i] = (T)(highL + (1.0 - mix) * lowL + mix * h);
            outR[i] = (T)(highR + (1.0 - mix) * lowR + mix * h);
        }
    }
};

// ------------------------------------------------------------- editor GUI ----
#if defined(_WIN32)
// Fixed editor size the host reads via effEditGetRect (top,left,bottom,right).
static VstRect g_edRect = { 0, 0, 200, 448 };

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
    ((PsychoBass*)e->object)->run<float>(in, out, n);
}
static void processDoubleReplacing(AEffect* e, double** in, double** out, VstInt32 n) {
    ((PsychoBass*)e->object)->run<double>(in, out, n);
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
    case effGetVendorVersion: return 1000;
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
    e->version     = 1000;
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
