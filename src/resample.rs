use dasp::{interpolate::sinc::Sinc, ring_buffer, signal, Sample, Signal};

pub fn resample(samples: &[f32], rate_from: u64, rate_to: u64) -> Vec<f32> {
    // Read the interleaved samples and convert them to a signal.
    let samples: Vec<f32> = samples
        .iter()
        .map(|sample| sample.to_sample::<f32>())
        .collect();

    let signal = signal::from_interleaved_samples_iter(samples);

    // Convert the signal's sample rate using `Sinc` interpolation.
    let ring_buffer = ring_buffer::Fixed::from([[0.0]; 100]);
    let sinc = Sinc::new(ring_buffer);
    let new_signal = signal.from_hz_to_hz(sinc, rate_from as f64, rate_to as f64);

    let output: Vec<f32> = new_signal
        .until_exhausted()
        .map(|frame| frame[0].to_sample::<f32>())
        .collect();

    output
}
