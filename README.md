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
| **Strength** (`Amount`) | 0–100% | Level of the added harmonics. 0% = no harmonics. |
| **Crossover** (`Xover`) | 40–400 Hz (log) | The split frequency. Below it: bass is isolated for harmonic synthesis and removed from the main path. Above it: untouched. |
| **Wet/Dry** (`Mix`) | 0–100% | 0% = original signal (tonally identical). 100% = real bass fully replaced by harmonics. |

> The GUI labels are abbreviated (`Amount`, `Xover`, `Mix`) because VST2 only
> guarantees an 8-byte buffer for parameter names (`kVstMaxParamStrLen`).

Signal flow: a Linkwitz-Riley 24 dB/oct crossover splits each channel into
phase-matched `low` + `high` bands → `bass = mono(low)` → envelope-normalize →
Chebyshev shaper (T2+T3+T4 → 2f/3f/4f at fixed relative levels, −6 dB per
step) → re-apply envelope → `HP + LP` (keep 2f–4f, cut above 4× crossover) →
`out = high + (1−Mix)·low + Mix·harmonics`. Only the low band crossfades, and
the LR4 bands are in phase, so the Wet/Dry sweep blends smoothly at every
setting — no crossover-region phase cancellation. The envelope normalization
keeps the harmonic recipe identical at every playback level (no fuzz on loud
passages, no dropout on quiet ones). Harmonic band-limiting is 24 dB/oct
Butterworth; all filters are RBJ biquads, processing runs in double precision,
denormals flushed.

## Build

No Steinberg SDK needed — `vst2_min.h` contains clean-room VST 2.4 definitions.

- **MinGW-w64:** `build_mingw.bat` (self-contained DLL, no runtime deps)
- **MSVC:** `build_msvc.bat` (needs the "Desktop development with C++" workload)

Output: `PsychoBass.dll` (64-bit).

## Install into Equalizer APO

1. Copy `PsychoBass.dll` to `C:\Program Files\EqualizerAPO\VSTPlugins\`
   (needs admin rights).
2. Add to your config (`config\config.txt` or a device-specific file):

   ```
   Plugin: -3dB
   VSTPlugin: PsychoBass.dll
   ```

   The `-3dB` preamp leaves headroom for the added harmonics.

   In the Equalizer APO **Configurator** GUI, the `VSTPlugin` block shows the
   three parameters as sliders — **Strength**, **Crossover**, **Mix** — because
   the plugin reports its parameter names and value displays to the host. Dial
   them in there; the Configurator writes the chosen values back into the config
   line for you.
3. In the Equalizer APO **Configurator**, make sure the effect is enabled on the
   playback device you want.

## Dialing it in

- **Crossover:** set it just above the frequency where your speakers/headphones
  actually give up — no higher. Everything below it gets replaced, so a
  too-high crossover throws away real bass your gear could have played;
  a too-low one leaves unreproducible fundamentals in the mix.
- **Wet/Dry:** high (80–100%) for tiny speakers that reproduce nothing below
  the crossover; moderate (30–60%) for headphones or speakers that manage
  *some* low end, so real bass and harmonics share the load.
- **Strength:** to taste, after the other two. It only sets how loud the
  synthesized harmonics are.
- Keep the `-3dB` preamp (or more at high Strength) — the added harmonics
  need headroom.

## Caveats (inherent to the technique, not bugs)

- **It's an illusion, not a subwoofer.** The perceived depth comes from your
  brain reconstructing the missing fundamental; it works remarkably well on
  melodic bass but can't recreate the physical punch of real low end.
- **Complex low material picks up some grit.** The waveshaper intermodulates
  simultaneous bass notes (e.g. 55+70 Hz also produces 125 Hz). The products
  stay in the bass register and mostly read as "bigger", but dense mixes get
  a slightly rougher low end at high Strength.
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
