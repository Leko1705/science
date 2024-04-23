package science.nn.layer;

import science.nn.functional.Function;
import science.nn.graph.Neuron;

import java.util.Objects;

public class ActivationLayer implements Layer {

    private final Function function;

    private Layer prevLayer;

    public ActivationLayer(Function function) {
        this.function = Objects.requireNonNull(function);
    }

    @Override
    public Kind getKind() {
        return prevLayer.getKind();
    }

    @Override
    public void forward() {
       for (Neuron neuron : prevLayer.getShape()){
           neuron.setValue(function.eval(neuron.getValue()));
       }
    }

    @Override
    public void backward() {
    }

    @Override
    public Shape getShape() {
        return prevLayer.getShape();
    }

    @Override
    public void connectPrevious(Layer layer) {
        prevLayer = layer;
    }

}
