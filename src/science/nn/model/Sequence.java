package science.nn.model;

import science.nn.graph.Neuron;
import science.nn.layer.DimensionMismatchException;
import science.nn.layer.Layer;
import science.nn.layer.Shape;
import science.nn.loss.Loss;
import science.nn.optim.Optimizer;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class Sequence implements Model {

    private final List<Layer> layers = new ArrayList<>();

    @Override
    public double[] generate(double[] in) {
        if (layers.isEmpty()) return in;

        Layer firstLayer = layers.get(0);
        fillLayer(firstLayer, in);

        Layer layer = firstLayer;
        for(int L = 1; L < layers.size(); L++) {
            layer = layers.get(L);
            layer.forward();
        }

        List<Double> outputs = new ArrayList<>();
        for (Neuron neuron : layer.getShape())
            outputs.add(neuron.getValue());
        return outputs.stream().mapToDouble(i -> i).toArray();
    }


    private void fillLayer(Layer layer, double[] in) {
        Shape firstShape = layer.getShape();
        Iterator<Neuron> neurons = firstShape.iterator();

        for (double v : in) {
            if (!neurons.hasNext()) throw new DimensionMismatchException("input size is too big");
            Neuron neuron = neurons.next();
            neuron.setValue(v);
        }
        if (neurons.hasNext()) throw new DimensionMismatchException("input size is too small");
    }

    @Override
    public void backward(Loss loss) {
        if (layers.isEmpty()) return;
        Layer outputLayer = layers.get(layers.size()-1);

        int i = 0;
        for (Neuron neuron : outputLayer.getShape()) {
            double lossGrad = loss.gradientAt(i++);
            double squashedGrad = neuron.squash(neuron.getUnsquashed());
            // add up
            neuron.setGradient(neuron.getGradient() + (lossGrad * squashedGrad));
        }

        for (int L = layers.size()-2; L >= 0; L--){
            Layer hiddenLayer = layers.get(L);
            hiddenLayer.backward();
        }
    }

    @Override
    public void fit(Optimizer optimizer) {
        for (int L = layers.size()-1; L >= 0; L--)
            optimizer.optimize(layers.get(L));
    }

    @Override
    public void zeroGradients() {
        for (Layer layer : layers) {
            for (Neuron neuron : layer.getShape())
                neuron.setGradient(0);
        }
    }

    public void addLayer(Layer layer) {
        if (layers.isEmpty())
            layers.add(layer);
        else
            addAndConnect(layer);
    }

    private void addAndConnect(Layer layer) {
        Layer lastLayer = layers.get(layers.size()-1);
        layer.connectPrevious(lastLayer);
        layers.add(layer);
    }

}
