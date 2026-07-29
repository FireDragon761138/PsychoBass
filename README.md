# PsychoBass — psychoacoustic bass enhancer (VST2, 64-bit)

A "virtual bass" plugin for Equalizer APO. It **removes the real low
frequencies** with a highpass, synthesizes their **harmonics** (2f, 3f, 4f…)
with a nonlinear waveshaper, band-limits them, and **blends them back** into the
stereo signal. Your ear reconstructs the *missing fundamental* from the
harmonics, so you hear deep bass that small speakers/headphones can't physically
reproduce — without the flabby low end.

## Controls

| Knob (GUI label) | Range | What it does |
|------------------|-------|--------------|
| **Strength** (`Amount`) | 0–100% | Level of the added harmonics, as a ratio against the bass being removed. 0% = no harmonics. |
| **Crossover** (`Xover`) | 40–400 Hz (log) | The split frequency. Below it: bass is isolated for harmonic synthesis and removed from the main path. Above it: untouched. |
| **Wet/Dry** (`Mix`) | 0–100% | 0% = original signal (tonally identical). 100% = real bass fully replaced by harmonics. |

**Defaults** resets all three to factory values — Strength 50%, Crossover
mid-travel (~126 Hz), Wet/Dry 75%.

Every parameter is a plain 0–1 value at the host interface; the GUI and the
Configurator show it as a percentage or as Hz. Crossover maps logarithmically,
`f = 40 · 10^p`, so the knob position for a frequency is `p = log10(f / 40)`:

| Hz | 40 | 50 | 60 | 80 | 100 | 120 | 150 | 200 | 300 | 400 |
|---|---|---|---|---|---|---|---|---|---|---|
| knob | 0.00 | 0.10 | 0.18 | 0.30 | 0.40 | 0.48 | 0.57 | 0.70 | 0.87 | 1.00 |

> The GUI labels are abbreviated (`Amount`, `Xover`, `Mix`) because VST2 only
> guarantees an 8-byte buffer for parameter names (`kVstMaxParamStrLen`).

## How it works

A Linkwitz-Riley 24 dB/oct crossover splits each channel into phase-matched
`low` + `high` bands. The low band is summed to mono, **split again into two
sub-bands**, and each sub-band is separately envelope-normalized, run through a
Chebyshev shaper (T2+T3+T4 → 2f/3f/4f at fixed relative levels, −6 dB per step),
and re-scaled by its envelope. The two harmonic streams are band-limited and
summed, level-matched against the bass being removed, then blended:

```
out = high + (1−Mix)·low + Mix·harmonics
```

Only the low band crossfades, and the LR4 bands are in phase, so the Wet/Dry
sweep blends smoothly at every setting — no crossover-region phase
cancellation.

**Why two sub-bands.** A memoryless nonlinearity fed two tones at once emits sum
and difference products belonging to neither (55 + 70 Hz also makes 125 Hz).
Splitting first means each shaper mostly sees one note. This is the refinement
that separates a serious virtual-bass implementation from a naive one, and it
measures: intermodulation products drop 4.7 dB (f1+f2) and 7.6 dB (f1+2·f2)
against the single-shaper design.

**Why per-band harmonic ceilings.** Each sub-band's harmonics stop two octaves
above *its own* top edge, not two octaves above the main crossover. A deep note
cannot produce a 4th harmonic above 4× the sub-split, so the old shared ceiling
merely let intermodulation and crossover-shoulder content spray an octave higher
than anything legitimate. Mid-band content drops ~2.5 dB across 240–480 Hz.

**Why loudness matching.** The synthesized level tracks the level of the bass
being removed, so **Strength sets a ratio** rather than a raw gain that drifts
with how much of the harmonic series survived the filters. Measured drift across
a 20 dB level change: 0.00 dB.

Harmonic band-limiting is 24 dB/oct Butterworth; all filters are RBJ biquads,
processing runs in double precision, denormals flushed.

## Build

No Steinberg SDK needed — `vst2_min.h` contains clean-room VST 2.4 definitions.

- **MinGW-w64:** `build_mingw.bat` → `PsychoBass.dll`
- **MSVC:** `build_msvc.bat` (needs the "Desktop development with C++" workload)

**One DLL runs everywhere.** The DSP kernel is written once and emitted twice —
baseline SSE2 and AVX2+FMA — with the plugin choosing between them at load time
from CPUID. There is no separate `-AVX2` build to install, nothing that can
fault on a pre-2013 CPU, and no second copy of the kernel to keep in sync: the
`vec2`/`vec4` types lower to two SSE2 instructions or one AVX2 instruction from
identical source. The editor footer shows which path is live.

The AVX2 emission costs **+10.7 KB** (67.1 KB vs 56.3 KB stripped) and is worth
**1.87×** on the audio kernel. `build_mingw.bat` greps the output for FMA
instructions, because if the kernel ever stops being inlined into its dispatch
wrapper the fast path vanishes *silently* — the plugin still works, it is just
never faster than SSE2. `build_mingw.bat baseline` builds SSE2-only for
comparison. The MSVC build is baseline-only.

## Cost

Measured at 48 kHz on dense low-frequency material, per audio sample:

| Build | ns/sample | × realtime | one core |
|---|---|---|---|
| AVX2 path (auto-selected) | 18.4 | 1131× | **0.088%** |
| SSE2 path | 28.7 | 726× | 0.137% |

The loop is throughput-bound on cascaded filter sections, so `-O3` changes
nothing (0.4%). What is left after that is packing: every stage of the harmonic
path that handles both sub-bands now handles them in one register pair instead
of two scalar registers, so the two bands cost about what one used to.

That is **19% off the AVX2 path** (22.9 → 18.4 ns) and **20% off SSE2**
(35.8 → 28.7 ns) against the previous build, from three changes:

- **The shaper runs paired.** The Chebyshev evaluation and its clamp and gate
  were two scalar calls whose results were then packed into a `vec2`; they are
  one `vec2` pass now. The operands arrive already packed from the sub-split, so
  the pairing costs no shuffling at either end.
- **The sub-band envelope followers run paired** — worth 9% on its own. The
  source used to argue against this on the grounds that the compare-and-select
  costs three SSE2 ops where the scalar form used one `cmov`. That is true per
  band, but it covers both bands, and the inputs were already sitting in a
  register pair. Measured rather than argued: 20.3 → 18.4 ns.
- **Settled knob smoothers stop ticking.** Two one-pole smoothers per sample
  exist to cover a knob in motion. Once they have closed on their targets they
  are constants, so they and the dry coefficient derived from them hoist out of
  the sample loop entirely. In a set-and-forget host that is every block.

Two things deliberately did **not** change. The loudness-match division still
runs at block rate, because it feeds a 50 ms smoother that cannot resolve the
difference. And nothing idles: an earlier plan to skip the harmonic branch
during silence, or when Mix or Strength sit at zero, was dropped before it was
written. Music and game audio always have signal in them, those knobs are never
zero in a real chain, and the saving would have been 0.1% of one core in a state
that does not occur — bought with a state machine in the hot loop and a new way
to click.

### Verifying that none of this changed the sound

`nulltest.exe <refDll> <newDll>` runs both builds over twelve scenarios — the
live preset, both Mix extremes, Strength zero, the two-tone case the sub-band
split exists for, material sitting inside and below the −70 dBFS harmonic gate,
silence, a loud/quiet slam that never lets the loudness match settle, noise, and
live knob automation — across six block sizes including 1, 3 and 17, so a
block-rate assumption cannot hide behind a convenient power of two. It reports
the largest sample difference on the float and double paths separately.

**The paired arithmetic is bit-identical**: every scenario, both paths, both
instruction sets. Confirmed by building with `-DPB_SNAP_EPS=0.0`, which disables
the smoother snap and leaves only the SIMD changes in play.

The shipped build differs from the previous one by at most **−249 dBFS** on the
double path, all of it the smoother snapping the last 1e-12 of its approach to
the target instead of creeping there over 21 seconds of denormal decay. On the
float path the same perturbation reads as up to **−157 dBFS**, which is one
float32 ULP — it is below the resolution of the format the samples are stored
in, so it can only appear where it tips a rounding boundary. The two paths get
different tolerances in the harness for exactly this reason, and it says so.

## Install into Equalizer APO

1. Copy `PsychoBass.dll` to your VST folder (`C:\Utilities\VST\`).
2. Add to your config:

   ```
   Plugin: -3dB
   VSTPlugin: PsychoBass.dll
   ```

   The `-3dB` preamp leaves headroom for the added harmonics.

   In the Equalizer APO **Configurator** GUI the `VSTPlugin` block shows the
   three parameters as sliders — **Strength**, **Crossover**, **Mix**. Dial them
   in there; the Configurator writes the chosen values back into the config line.
3. Make sure the effect is enabled on the playback device you want.

**Chain position matters.** PsychoBass goes *before* the headphone EQ, not
after. The EQ is a transducer correction, so everything that generates audible
content must sit upstream of it — otherwise the synthesized harmonics land on
the driver uncorrected, which is exactly where a mid-band smear comes from. The
protective highpass goes *after* PsychoBass: turning inaudible 20 Hz into
audible 60/80 Hz is the whole function, so highpassing first starves it.

## Dialing it in

**Start here**, then adjust by ear:

| Control | Start at | Why |
|---|---|---|
| **Wet/Dry** (`Mix`) | **0.75** (75%) | Most of the low band is harmonics, but real bass still carries what your gear can actually play. |
| **Strength** (`Amount`) | **0.5** (50%) | Half-ratio against the removed bass — audible without dominating the midrange. |
| **Crossover** (`Xover`) | **where your speakers or headphones start to fall off** | Everything below it is replaced; matching the rolloff means you only synthesize what you couldn't hear anyway. |

Strength is also called Amount — the GUI knob and the host parameter name for the
same control. Crossover has no universal starting number because it is a
property of your gear, not of the material: find the frequency where your
output stops delivering and set it there (see the Hz ↔ knob table above).

Refining from that point:

- **Crossover:** set it just above the frequency where your speakers/headphones
  actually give up — no higher. Everything below it gets replaced, so a
  too-high crossover throws away real bass your gear could have played;
  a too-low one leaves unreproducible fundamentals in the mix.
- **Wet/Dry:** raise toward 100% for tiny speakers that reproduce nothing below
  the crossover; drop toward 30–60% for headphones or speakers that manage
  *some* low end, so real bass and harmonics share the load. 75% suits most
  headphones.
- **Strength:** to taste, after the other two. Above ~70% the harmonics start
  competing with the mids rather than implying bass under them.
- Keep the `-3dB` preamp (or more at high Strength) — the added harmonics
  need headroom.

## Caveats (inherent to the technique, not bugs)

- **It's an illusion, not a subwoofer.** The perceived depth comes from your
  brain reconstructing the missing fundamental; it works remarkably well on
  melodic bass but can't recreate the physical punch of real low end.
- **Closely-spaced notes still intermodulate.** Two sub-bands separate a kick
  from a bass note roughly an octave above it, but notes a fourth or fifth
  apart land in the same band and still produce sum products. Fixing that
  needs more bands or a phase-vocoder analysis, at real CPU cost.
- **Mix = 0% is tonally, not bit-, transparent** — see Notes below.
- **Harmonics are mono.** Deliberate (bass is near-mono anyway), but any
  wide-stereo low-frequency content is narrowed in the harmonic image.

## Notes

- 64-bit only, matching Equalizer APO's audio pipeline on 64-bit Windows.
- At Mix = 0% the output is the recombined band split: identical magnitude at
  every frequency, with the usual LR4 phase rotation near the crossover
  (inaudible, and the standard trade of any zero-latency IIR crossover).
- Bass is summed to mono before synthesis (bass is near-mono anyway, and this
  avoids stereo artifacts); the harmonics are added equally to L and R.
- `test_host.exe [dll]` runs the behavioural checks and prints the
  intermodulation, mid-band spray and level-drift figures, so a change that
  regresses them is visible. It takes a DLL path so builds can be A/B'd.
- `nulltest.exe <refDll> <newDll>` differences two builds sample-for-sample.
  Any change that claims to be arithmetically neutral has to pass it.
- `bench.exe [dll]` reports per-sample cost. It is a design tool, not a ship
  gate.

## License

BSD 2-Clause. See [LICENSE](LICENSE).
