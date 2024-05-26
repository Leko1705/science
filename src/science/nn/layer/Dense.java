package science.nn.layer;

import science.nn.functional.Function;
import science.nn.graph.Weight;
import science.nn.graph.Neuron;
import science.nn.graph.SimpleNeuron;
import science.nn.graph.Weights;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class Dense implements Layer, Shape {

    private final double bias;

    private final List<Neuron> neurons = new ArrayList<>();

    public Dense(int size, Function function, double bias) {
        this.bias = bias;

        for (int i = 0; i < size; i++) {
            neurons.add(new SimpleNeuron(function));
        }
    }

    @Override
    public Kind getKind() {
        return Kind.LINEAR;
    }

    @Override
    public void forward() {
        for (Neuron neuron : neurons) {
            double sum = 0;
            for (Neuron pred : neuron.getPredecessors().keySet()) {
                Weight weight = neuron.getPredecessors().get(pred);
                sum += pred.getValue() * weight.get();
            }
            neuron.setValue(neuron.squash(sum + bias));
        }
    }

    @Override
    public void backward() {
        for (Neuron neuron : neurons){
            double sum = 0.0;

            for (Neuron successor : neuron.getSuccessors().keySet()){
                double previousGradient = successor.getGradient();
                double successorConnection = neuron.getSuccessors().get(successor).get();
                sum += previousGradient * successorConnection;
            }

            sum *= neuron.derivative(neuron.getUnsquashed());
            neuron.setGradient(neuron.getGradient() + sum);
        }
    }

    @Override
    public Shape getShape() {
        return this;
    }

    @Override
    public void connectPrevious(Layer layer) {
        for (Neuron neuron : layer.getShape()){
            for (Neuron next : neurons) {
                Weight weight = Weights.randomWeight();
                neuron.getSuccessors().put(next, weight);
                next.getPredecessors().put(neuron, weight);
            }
        }
    }

    @Override
    public int size() {
        return neurons.size();
    }

    @Override
    public Neuron get(int x) {
        return neurons.get(x);
    }

    @Override
    public void set(Neuron neuron, int x) {
        neurons.set(x, neuron);
    }

    @Override
    public Neuron[] toArray() {
        return neurons.toArray(new Neuron[0]);
    }

    @Override
    public Iterator<Neuron> iterator() {
        return neurons.iterator();
    }
}
