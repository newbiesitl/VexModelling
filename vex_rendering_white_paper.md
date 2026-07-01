# Vex Rendering: Learning the Acoustic Behavior of Guitars from Partial Sensor Observations

## Abstract

Acoustic guitars are complex mechanical-acoustic systems whose perceived sound is shaped not only by string vibration, but also by body geometry, soundboard response, bracing structure, tonewoods, air-cavity resonance, pickup placement, and the downstream pickup–preamp signal chain. Contemporary acoustic guitar amplification systems rely on magnetic soundhole pickups, piezo pickups, analog preamps, digital filters, impulse-response processing, and acoustic imaging techniques to approximate microphone-like tone. However, these systems remain constrained by pickup-specific coloration, preamp pairing, limited controllability, and incomplete observation of the full radiated acoustic field.

This paper introduces **Vex Rendering** as an ongoing research project for learning the acoustic behavior of acoustic guitars from partial sensor observations. The central hypothesis is that pickup and contact-sensor signals should be treated as incomplete measurements of a hidden acoustic state, while microphone recordings serve as target observations of the instrument’s radiated sound. As a first-stage baseline, we study a HiFi-GAN-based direct neural rendering approach that maps pickup-derived signals to microphone-like acoustic audio. This baseline provides a practical proof of concept for data-driven acoustic rendering, but also exposes limitations of non-latent black-box generation, including weak physical interpretability, limited controllability, pickup-specific overfitting, and difficulty representing body resonance and spatial acoustic behavior.

Motivated by these limitations, we propose a DDSP-inspired latent-variable direction for structured acoustic modeling. Instead of directly translating pickup signals into waveforms, the proposed framework infers intermediate acoustic variables such as pitch, harmonic energy, transient excitation, decay envelopes, resonance behavior, and noise components, then renders audio through differentiable synthesis modules. This formulation aims to provide stronger physical inductive bias, better interpretability, and a more flexible path toward controllable acoustic sound generation. More broadly, Vex Rendering serves as a research probe into whether modern neural architectures can learn, represent, and eventually manipulate the acoustic behavior of instruments from partial observations.


# 1. Introduction

The acoustic guitar occupies an important position in both musical culture and the musical-instrument industry. As a portable, expressive, and harmonically rich instrument, it is widely used across folk, pop, rock, country, jazz, fingerstyle, and contemporary instrumental music. The guitar market remains a significant part of the global instrument economy; recent market reports estimate the global guitar market in the range of several billions to more than ten billion U.S. dollars, with acoustic guitars representing a major product segment (see, for example, Grand View Research, *Guitar Market Size Report, 2023*, available at https://www.grandviewresearch.com/industry-analysis/guitar-market). Beyond its commercial importance, the acoustic guitar is also a compelling physical system: a small mechanical excitation at the string is transformed by the bridge, soundboard, bracing, back, sides, internal air cavity, and surrounding environment into a complex radiated acoustic field.

This complexity makes acoustic guitar amplification and modeling a difficult problem. Unlike an electric guitar, where the pickup signal is often treated as the primary musical signal, the perceived sound of an acoustic guitar is strongly shaped by the instrument body itself. The strings, saddle, bridge, soundboard, bracing pattern, tonewoods, body geometry, and air resonance jointly determine how mechanical energy is distributed and radiated. A dreadnought guitar, an OM guitar, a small-body parlor guitar, and a modern lattice-braced instrument may respond very differently to the same playing gesture. Therefore, the microphone sound of an acoustic guitar is not merely the sound of vibrating strings; it is the result of a coupled mechanical-acoustic rendering process.

Contemporary acoustic guitar pickup systems add another layer of complexity. A magnetic soundhole pickup, an undersaddle piezo pickup, a bridge-plate transducer, and an internal microphone each observe different partial projections of the instrument. A soundhole magnetic pickup can produce a strong, musical, and feedback-resistant signal, especially when paired with a suitable preamp, but it primarily senses string motion and may not preserve the full acoustic body response. A piezo pickup is more directly coupled to bridge or saddle vibration and can capture attack and mechanical detail, but it is still a local sensor and often lacks the spatial radiation, air coupling, and body image captured by an external microphone. In practical performance systems, the final tone is further shaped by pickup placement, impedance matching, preamp design, gain staging, EQ, compression, phase alignment, reverb, and blending. Thus, acoustic guitar amplification is already a hand-designed rendering chain rather than a transparent reproduction of the instrument.

Existing commercial solutions approach this problem using a mixture of analog circuitry, digital filtering, impulse response processing, microphone modeling, and acoustic imaging. Traditional analog systems include tube-based and transistor-based preamps, each introducing its own tonal coloration and impedance behavior. Digital systems such as Fishman Aura, LR Baggs Voiceprint DI, Audio Sprockets ToneDexter, BOSS AD-10, and Yamaha AG-Stomp attempt to recover a more microphone-like acoustic tone through impulse responses, learned pickup-to-microphone transforms, acoustic resonance controls, or microphone modeling (Fishman Aura; LR Baggs Voiceprint DI; ToneDexter; BOSS AD-10; Yamaha AG-Stomp). These systems demonstrate that the industry has long recognized the gap between pickup sound and microphone sound. However, they remain limited by pickup-preamp pairing, fixed processing structures, handcrafted controls, limited model capacity, and difficulty representing the full nonlinear, time-varying behavior of the instrument.

Recent progress in deep generative audio models suggests a different research direction. Models such as WaveNet and HiFi-GAN have shown that neural networks can generate high-fidelity waveforms and learn complex audio distributions directly from data (WaveNet; HiFi-GAN). Although these methods were originally developed mainly for speech synthesis and neural vocoding, they motivate a broader question: can similar data-driven methods be used to learn the acoustic behavior of musical instruments from partial sensor observations? Instead of manually designing an acoustic rendering chain through pickup selection, preamp coloration, EQ, and DSP filters, we may train neural models to infer the missing acoustic structure between a local pickup signal and a microphone-like radiated sound.

This paper introduces **Vex Rendering** as an ongoing research project toward neural acoustic modeling of acoustic guitars. The goal is not only to make a pickup signal sound more pleasing, but to study whether modern neural architectures can learn the relationship between partial sensor observations and the fuller acoustic behavior of the instrument. In this sense, Vex Rendering treats pickup signals as incomplete measurements of a hidden acoustic state, and microphone recordings as target observations of the radiated acoustic field.

As a first step, this work studies a direct neural rendering baseline based on HiFi-GAN. The baseline tests whether a high-capacity adversarial waveform generator can map pickup-derived features to microphone-like acoustic guitar audio. This approach is useful because it provides an initial proof of concept and exposes the strengths and weaknesses of purely data-driven, non-latent-variable rendering. However, direct neural rendering does not explicitly model the physical or perceptual factors that produce acoustic guitar tone. It may learn a plausible mapping from input to output, but it does not necessarily reveal how string excitation, body resonance, transient noise, decay behavior, or spatial radiation are represented.

Motivated by these limitations, the paper then proposes a DDSP-inspired latent-variable research direction. Instead of learning only a black-box mapping from pickup signal to waveform, the latent-variable formulation introduces an intermediate acoustic state that can represent pitch, harmonic energy, transient components, resonance envelopes, body response, and noise-like excitation. A structured differentiable renderer can then synthesize the target waveform from these inferred latent variables. This approach is intended to provide stronger physical inductive bias, better interpretability, and more controllable acoustic rendering.

The purpose of this paper is therefore exploratory. It documents the first stage of Vex Rendering as a research probe into neural acoustic guitar modeling: beginning with a HiFi-GAN baseline, analyzing the limitations of direct non-latent rendering, and outlining a DDSP-style latent-variable path toward more interpretable acoustic modeling. More broadly, this work asks whether the complex acoustic behavior of an instrument can be learned from data, represented through structured latent variables, and eventually manipulated to produce realistic, personalized, or even novel acoustic timbres.





## Project Status

This repository documents an ongoing research project. The current stage focuses on:
- establishing a HiFi-GAN-based direct neural rendering baseline;
- analyzing the limitations of non-latent pickup-to-microphone rendering;
- developing a DDSP-inspired latent-variable framework for interpretable acoustic modeling.

## Paper Draft

The working paper is organized as:
1. Introduction
2. Acoustic Guitar System Background
3. Problem Formulation
4. Related Work
5. Vex Rendering System Overview
6. HiFi-GAN Baseline
7. Limitations of Direct Neural Rendering
8. DDSP-Based Latent Acoustic Modeling
9. Workflow Comparison
10. Experimental Design
11. Discussion
12. Limitations and Future Work
13. Conclusion