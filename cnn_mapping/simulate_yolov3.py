
class Layer:
    def __init__(self, type, input_layer, in_size, out_size, f_size, op1, op2, op3):
        self.type = type
        self.input = input_layer
        self.in_size = in_size
        self.out_size = out_size
        self.f_size = f_size
        self.op1 = op1
        self.op2 = op2
        self.op3 = op3
    
    def set_pe(self, pe):
        self.pe = pe
    
    def set_out_layer(self, out_layer):
        self.out_layer = out_layer

class ConvLayer(Layer):
    def __init__(self, input_layer, cout, hf, wf):
        self.cout = cout
        self.hf = hf
        self.wf = wf
        self.stride = 1
        self.hout = self.hin = input_layer.hout
        self.wout = self.win = input_layer.wout
        self.cin = input_layer.cout
        
        fsize = self.hf * self.wf * self.cin * self.cout
        out_size = self.hout * self.wout * self.cout
        
        Layer.__init__(self, 'conv', input_layer,
            self.hin * self.win * self.cin,
            out_size,
            fsize,
            self.hin * self.win * fsize,
            out_size * (self.hf * self.wf * self.cin - 1),
            0,
        )

class MaxPoolLayer(Layer):
    def __init__(self, input_layer, hf, wf, stride):
        self.cout = input_layer.cout
        self.hf = hf
        self.wf = wf
        self.stride = stride
        self.hin = input_layer.hout
        self.win = input_layer.wout
        self.cin = input_layer.cout
        self.hout = self.hin / stride
        self.wout = self.win / stride

        in_size = in_size = self.hin * self.win * self.cin

        Layer.__init__(self, 'maxp', input_layer,
            in_size,
            self.hout * self.wout * self.cout,
            0,
            0,
            in_size * (self.hf * self.wf - 1) / self.stride,
            in_size / self.stride
        )

class RouteLayer(Layer):
    def __init__(self, input_layers):
        in_size = 0
        self.cout = 0
        self.input_layers = input_layers
        for layer in input_layers:
            in_size += layer.out_size
            self.cout += layer.cout

        self.hout = input_layers[0].hout / len(input_layers)
        self.wout = input_layers[0].wout / len(input_layers)

        Layer.__init__(self, 'route', input_layers[-1],
            in_size,
            self.hout * self.wout * self.cout,
            0,
            0, 0, 0
        )

class UpSamplingLayer(Layer):
    def __init__(self, input_layer, hf, wf):
        self.hout = input_layer.hout * hf
        self.wout = input_layer.wout * wf
        self.cout = input_layer.cout
        Layer.__init__(self, 'upsample', input_layer,
            input_layer.out_size,
            self.hout * self.wout * self.cout,
            0,
            0,0,0
        )

class InputLayer(Layer):
    def __init__(self, w, h, c):
        self.cout = c
        self.hout = h
        self.wout = w
        Layer.__init__(self, 'input', None, w*h*c, w*h*c, 0, 0, 0, 0)

class YoloV3TinyRuntime:
    def __init__(self):
        inputs = InputLayer(416, 416, 3)
        yolov3 = []
        # 0
        yolov3.append(ConvLayer(inputs, 16, 3, 3))
        # 1
        yolov3.append(MaxPoolLayer(yolov3[-1], 2, 2, 2))
        # 2
        yolov3.append(ConvLayer(yolov3[-1], 32, 3, 3))
        # 3
        yolov3.append(MaxPoolLayer(yolov3[-1], 2, 2, 2))
        # 4
        yolov3.append(ConvLayer(yolov3[-1], 64, 3, 3))
        # 5
        yolov3.append(MaxPoolLayer(yolov3[-1], 2, 2, 2))
        # 6
        yolov3.append(ConvLayer(yolov3[-1], 128, 3, 3))
        # 7
        yolov3.append(MaxPoolLayer(yolov3[-1], 2, 2, 2))
        # 8
        yolov3.append(ConvLayer(yolov3[-1], 256, 3, 3))
        # 9
        yolov3.append(MaxPoolLayer(yolov3[-1], 2, 2, 2))
        # 10
        yolov3.append(ConvLayer(yolov3[-1], 512, 3, 3))
        # 11
        yolov3.append(MaxPoolLayer(yolov3[-1], 2, 2, 1))
        # 12
        yolov3.append(ConvLayer(yolov3[-1], 1024, 3, 3))
        # 13
        yolov3.append(ConvLayer(yolov3[-1], 256, 1, 1))
        # 14
        yolov3.append(ConvLayer(yolov3[-1], 512, 3, 3))
        # 15
        yolov3.append(ConvLayer(yolov3[-1], 255, 1, 1))
        # 16
        yolov3.append(Layer('yolo', yolov3[-1], yolov3[-1].out_size, 13*13*256, 0, 0,0,0))
        # 17
        yolov3.append(RouteLayer([yolov3[13]]))
        # 18
        yolov3.append(ConvLayer(yolov3[-1], 128, 1, 1))
        # 19
        yolov3.append(UpSamplingLayer(yolov3[-1], 2, 2))
        # 20
        yolov3.append(RouteLayer([yolov3[8], yolov3[19]]))
        # 21
        yolov3.append(ConvLayer(yolov3[-1], 256, 3, 3))
        # 22
        yolov3.append(ConvLayer(yolov3[-1], 255, 1, 1))
        # 23
        yolov3.append(Layer('yolo', yolov3[-1], yolov3[-1].out_size, 13*13*256, 0, 0,0,0))

        self.inputs = inputs
        self.layers = yolov3

        self.inputs.set_out_layer(self.layers[0])
        for i in range(len(self.layers) - 1):
            self.layers[i].set_out_layer(self.layers[i+1])

        self.max_time = self._get_max_time()

    def _get_layer_comp_time(self, layer, pe):
        if layer.type == 'conv' or layer.type == 'maxp':
            if pe == 0:
                comp_time = ((7 * layer.op1) + (3 * layer.op2) + (30 * layer.op3) + 40 * (layer.in_size + layer.f_size) + layer.out_size) / ( 1e8)
            else:
                comp_time = (layer.op1 + layer.op2 + layer.op3 + layer.in_size + layer.out_size) / 32 / 50e6
        else:
            comp_time = 0
        return comp_time
    
    def _get_layer_comm_time(self, layer, prevpe, pe):
        comm_time = 0
        if layer.type == 'route':
            pe = 0
            for rl in layer.input_layers:
                if rl.pe == 1 and rl.out_layer.pe == 1:
                    comm_time += rl.out_size / (1 << 13)
        else:
            if layer.type == 'conv' or layer.type == 'maxp':
                if pe == 1:
                    comm_time += layer.f_size / (1 << 15)
            else:
                pe = 0
            if prevpe != pe:
                comm_time += layer.in_size / (1 << 15)
        return comm_time

    def _get_max_time(self):
        comp_time = 0
        comm_time = 0
        for layer in self.layers:
            layer.set_pe(1)
            comp_time += max(self._get_layer_comp_time(layer, 0), self._get_layer_comp_time(layer, 1))
            comm_time += max(self._get_layer_comm_time(layer, 1, 0), self._get_layer_comm_time(layer, 0, 1))
        return comp_time + comm_time

    def get_run_time(self, pe_mapping):
        prevpe = 0
        comm_time = 0
        comp_time = 0
        for pe, layer in zip(pe_mapping, self.layers):
            if not (layer.type == 'conv' or layer.type == 'maxp'):
                pe = 0
            layer.set_pe(pe)
            comm_time += self._get_layer_comm_time(layer, prevpe, pe)
            comp_time += self._get_layer_comp_time(layer, pe)
            prevpe = pe

        runtime = comp_time + comm_time
        return runtime
    



def main():
    yolo = YoloV3TinyRuntime()


    print('max time', yolo.max_time)

if __name__ == '__main__':
    main()