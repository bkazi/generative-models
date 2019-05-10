import tensorflow as tf


def real_nvp_template(hidden_layers,
                      shift_only=False,
                      activation=tf.nn.leaky_relu,
                      extra_info=None,
                      training=True,
                      use_batch_norm=False,
                      use_drop_out=False,
                      regularizer=None,
                      initializer=None,
                      name=None,
                      *args,
                      **kwargs):

    with tf.name_scope(name, "real_nvp_template"):

        def _fn(x, output_units, **condition_kwargs):
            if condition_kwargs:
                raise NotImplementedError(
                    "Conditioning not implemented in the default template.")

            if tf.rank(x) == 1:
                x = x[tf.newaxis, ...]
                def reshape_output(x): return x[0]
            else:
                def reshape_output(x): return x

            if extra_info is not None:
                e_x = extra_info()
                x = tf.concat([x, e_x], axis=-1)

            for i, units in enumerate(hidden_layers):
                x = tf.layers.dense(
                    inputs=x,
                    units=units,
                    activation=None,
                    kernel_regularizer=regularizer,
                    kernel_initializer=initializer,
                    *args,
                    **kwargs
                )
                if use_batch_norm and i % 2 == 0:
                    x = tf.layers.batch_normalization(
                        x,
                        gamma_initializer=initializer,
                        beta_initializer=initializer,
                        training=training,
                        *args,
                        **kwargs
                    )
                x = activation(x)
                if use_drop_out and i % 2 == 0:
                    x = tf.layers.dropout(
                        x,
                        rate=0.4,
                        training=training
                    )
            x = tf.layers.dense(
                inputs=x,
                units=(1 if shift_only else 2) * output_units,
                activation=None,
                kernel_regularizer=regularizer,
                kernel_initializer=tf.initializers.zeros(),
                *args,
                **kwargs
            )
            if shift_only:
                return reshape_output(x), None
            shift, log_scale = tf.split(x, 2, axis=-1)
            # log_scale = tf.nn.sigmoid(log_scale)
            return reshape_output(shift), reshape_output(log_scale)

        return tf.make_template("real_nvp_template", _fn)
