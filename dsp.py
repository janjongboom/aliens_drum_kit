from jax import grad, jit
from jax.experimental import jax2tf
import jax.numpy as jnp
import tensorflow as tf
import numpy as np
import os
import math
from spectrogram.mfe import generate_features as mfe_features
from graphing import create_mfe_graph

def generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq,
                      frame_length, frame_stride, num_filters, fft_length,
                      low_frequency, high_frequency, noise_floor_db):
    if (num_filters < 2):
        raise Exception('Filter number should be at least 2')
    if (not math.log2(fft_length).is_integer()):
        raise Exception('FFT length must be a power of 2')
    if (len(axes) != 1):
        raise Exception('MFE blocks only support a single axis, ' +
            'create one MFE block per axis under **Create impulse**')

    features = mfe_features(signal=jnp.array(raw_data),
                            sampling_freq=sampling_freq,
                            frame_length=frame_length,
                            frame_stride=frame_stride,
                            num_filters=num_filters,
                            fft_length=fft_length,
                            low_frequency=low_frequency,
                            high_frequency=high_frequency,
                            noise_floor_db=noise_floor_db)

    width = np.shape(features)[0]
    height = np.shape(features)[1]

    graphs = []

    if draw_graphs:
        # hardcoded for now
        filterbank_freqs = [  44.37407701,   91.56109503,  141.73937073,  195.09852453,  251.84019719,
                             312.17881177,  376.34238398,  444.57338374,  517.12965156,  594.28537283,
                             676.33211398,  763.57992429,  856.35850754,  955.01846792, 1059.93263499,
                             1171.49747253, 1290.13457677, 1416.29226959, 1550.44729265, 1693.10660904,
                             1844.80931898, 2006.12869712, 2177.67435884, 2360.094564,   2554.07866662,
                             2760.35971998, 2979.71724669, 3212.98018452, 3461.03001887, 3724.80411386,
                             4005.29925458, 4303.57541383, 4620.75975774, 4958.05090523, 5316.72345751,
                             5698.13281472, 6103.72029792, 6535.01859571, 6993.65755619, 7481.37034603 ]

        image = create_mfe_graph(
            sampling_freq, frame_length, frame_stride, width, height, np.swapaxes(features, 0, 1), filterbank_freqs)

        graphs.append({
            'name': 'Mel Filterbank Energies',
            'image': image,
            'imageMimeType': 'image/svg+xml',
            'type': 'image'
        })

    return {
        'features': features.flatten().tolist(),
        'graphs': graphs,
        'fft_used': [ fft_length ],
        'output_config': {
            'type': 'spectrogram',
            'shape': {
                'width': width,
                'height': height
            }
        }
    }

def get_dsp_impl(sampling_freq,
                 frame_length, frame_stride, num_filters, fft_length,
                 low_frequency, high_frequency, noise_floor_db):
    def get_features(raw_data):
        return mfe_features(signal=raw_data,
                            sampling_freq=sampling_freq,
                            frame_length=frame_length,
                            frame_stride=frame_stride,
                            num_filters=num_filters,
                            fft_length=fft_length,
                            low_frequency=low_frequency,
                            high_frequency=high_frequency,
                            noise_floor_db=noise_floor_db)
    return get_features

# Get the corresponding TFLite model that maps to generate_features w/ the same parameters
def get_tflite_implementation(implementation_version, input_shape, axes, sampling_freq,
                              frame_length, frame_stride, num_filters, fft_length,
                              low_frequency, high_frequency, noise_floor_db):
    get_features_fn = get_dsp_impl(sampling_freq,
                                   frame_length, frame_stride, num_filters, fft_length,
                                   low_frequency, high_frequency, noise_floor_db)

    tf_predict = tf.function(
        jax2tf.convert(get_features_fn, enable_xla=False),
        input_signature=[
            tf.TensorSpec(shape=input_shape, dtype=tf.float32, name='input')
        ],
        autograph=False)
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [tf_predict.get_concrete_function()], tf_predict)

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        # tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
    ]
    tflite_float_model = converter.convert()
    return tflite_float_model
