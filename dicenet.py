from models.utils import get_required_argument
import tensorflow as tf


class DICENET:
    def __init__(self, params):
        self.name = get_required_argument(params, 'name', 'Must provide name.')
        self.discount = params.get('discount')
        self.lam = params.get('lambda_param')
        self.xi = params.get('xi_param')

        if params.get('sess', None) is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self._sess = tf.Session(config=config)
        else:
            self._sess = params.get('sess')
        self.num_nets = params.get('num_networks', 1)
        self.ratio_layers = []
        self.f_function_layers = []

        self.ratio_decays = []
        self.f_function_decays = []
        self.f_u_vars = []
        self.tau_vars = []

        self.u = None
        self.optimizer_f_u = None
        self.optimizer_tau = None
        self.s_a_input = None
        self.next_s_a_input = None
        self.init_s_a_input = None
        self.train_op = None
        self.pred_sa_input = None
        self.predicted_ratio = None
        self.tau_loss = None

    def add_ratio_net(self, layer):
        layer.set_ensemble_size(self.num_nets)
        if len(self.ratio_layers) > 0:
            layer.set_input_dim(self.ratio_layers[-1].get_output_dim())
        self.ratio_layers.append(layer.copy())

    def add_f_function_net(self, layer):
        layer.set_ensemble_size(self.num_nets)
        if len(self.f_function_layers) > 0:
            layer.set_input_dim(self.f_function_layers[-1].get_output_dim())
        self.f_function_layers.append(layer.copy())

    def finalize(self, optimizer, optimizer_args=None):
        optimizer_args = {} if optimizer_args is None else optimizer_args
        self.optimizer_f_u = optimizer(**optimizer_args)
        self.optimizer_tau = optimizer(**optimizer_args)

        # construct all variables
        with self._sess.as_default():
            with tf.variable_scope(self.name):
                self.u = tf.Variable(tf.constant(0.0, shape=[self.num_nets, 1]))
                self.f_u_vars.append(self.u)
                for i, layer in enumerate(self.ratio_layers):
                    with tf.variable_scope("ratio_Layer%i" % i):
                        layer.construct_vars()
                        self.ratio_decays.extend(layer.get_decays())
                        self.tau_vars.extend(layer.get_vars())
                for i, layer in enumerate(self.f_function_layers):
                    with tf.variable_scope("f_function_Layer%i" % i):
                        layer.construct_vars()
                        self.f_function_decays.extend(layer.get_decays())
                        self.f_u_vars.extend(layer.get_vars())

        # set up training
        self.s_a_input = tf.placeholder(dtype=tf.float32,
                                        shape=[self.num_nets, None, self.ratio_layers[0].get_input_dim()])
        self.next_s_a_input = tf.placeholder(dtype=tf.float32,
                                             shape=[self.num_nets, None, self.ratio_layers[0].get_input_dim()])
        self.init_s_a_input = tf.placeholder(dtype=tf.float32,
                                             shape=[self.num_nets, None, self.ratio_layers[0].get_input_dim()])
        tau = self._compile_ratio_outputs(self.s_a_input)
        self.tau = tau
        f = self._compute_f_function_outputs(self.s_a_input)
        f_next = self._compute_f_function_outputs(self.next_s_a_input)
        f_0 = self._compute_f_function_outputs(self.init_s_a_input)

        # loss = (1 - self.discount) * f_0 + self.discount * tau * f_next - tau * f - 0.5 * tf.square(f) + self.lam * (self.u[:, None, :] * tau - self.u[:, None, :] - 0.5 * tf.square(self.u[:, None, :])) 
            
        # f_function_loss = loss
        # final_f_function_loss = - tf.reduce_mean(f_function_loss)
        # self.f_function_loss = final_f_function_loss
        # train_f_u_op = self.optimizer_f_u.minimize(final_f_function_loss, var_list=self.f_u_vars)

        # tau_loss = loss
        # self.tau_loss = tf.reduce_mean(tau_loss)
        # final_tau_loss = self.tau_loss
        # + tf.add_n(self.ratio_decays)

        f_function_loss = (1 - self.discount) * f_0 + self.discount * tau * f_next - tau * f - 0.5 * tf.square(
            f) + self.lam * (self.u[:, None, :] * tau - self.u[:, None, :] - 0.5 * tf.square(self.u)[:, None, :])
        final_f_function_loss = - tf.reduce_mean(f_function_loss)
        train_f_u_op = self.optimizer_f_u.minimize(final_f_function_loss, var_list=self.f_u_vars)
        tau_loss = self.discount * tau * f_next - tau * f + self.lam * (self.u[:, None, :] * tau)
        self.tau_loss = tf.reduce_mean(tau_loss)
        final_tau_loss = self.tau_loss
        # + tf.add_n(self.ratio_decays)

        train_tau_op = self.optimizer_tau.minimize(final_tau_loss, var_list=self.tau_vars)
        self.train_op = tf.group(train_f_u_op, train_tau_op)

        # initialize all variables
        self._sess.run(tf.variables_initializer(
            self.tau_vars + self.f_u_vars + self.optimizer_tau.variables() + self.optimizer_f_u.variables()))

        # set up prediction
        self.pred_sa_input = tf.placeholder(dtype=tf.float32,
                                            shape=[None, self.ratio_layers[0].get_input_dim()])
        self.predicted_ratio = self._compile_ratio_outputs(self.pred_sa_input)

    # train
    def train_dicenet(self, sa_input, next_sa_input, init_sa_input):
        _, ratio_losses = self._sess.run([self.train_op, self.tau_loss],
                                         feed_dict={self.s_a_input: sa_input, self.next_s_a_input: next_sa_input,
                                                    self.init_s_a_input: init_sa_input})
        # print('dicenet_loss:', f_function_loss)
        # print('tau', tau)
        return ratio_losses

    # predict
    def predict_ratio(self, sa_input):
        return self._sess.run(self.predicted_ratio, feed_dict={self.pred_sa_input: sa_input})

    # compile
    def _compile_ratio_outputs(self, sa_input):
        cur_out = sa_input
        for layer in self.ratio_layers:
            cur_out = layer.compute_output_tensor(cur_out)
        return cur_out

    def _compute_f_function_outputs(self, sa_input):
        cur_out = sa_input
        for layer in self.f_function_layers:
            cur_out = layer.compute_output_tensor(cur_out)
        return cur_out
