package science.nn.layer;

import science.nn.functional.Function;
import science.nn.functional.Identity;
import science.nn.graph.Neuron;
import science.nn.graph.SimpleNeuron;
import science.nn.graph.Weight;
import science.nn.graph.Weights;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class PollingLayer1D implements Layer, Shape {

    private static final Function IDENTITY = new Identity();


    private final Poller1D poller;

    private final int filterSpan;

    private final List<Neuron> filteredMap = new ArrayList<>();


    public PollingLayer1D(Poller1D poller, int filterSpan) {
        this.poller = poller;
        this.filterSpan = filterSpan;
    }

    @Override
    public Kind getKind() {
        return Kind.LINEAR;
    }

    @Override
    public void forward() {
        for (Neuron neuron : filteredMap) {
            Neuron[] group = neuron.getPredecessors().keySet().toArray(new Neuron[0]);
            double polled = poller.poll(group);
            neuron.setValue(polled);
        }
    }

    @Override
    public void backward() {
        for (Neuron n : filteredMap) {
            n.setGradient(0);
        }
    }

    @Override
    public Shape getShape() {
        return this;
    }

    @Override
    public void connectPrevious(Layer layer) {
        Neuron[] neurons = layer.getShape().toArray();

        int outSize = neurons.length - filterSpan + 1;

        for (int i = 0; i < outSize; i++) {
            Neuron filteredNeuron = new SimpleNeuron(IDENTITY);

            for (int j = 0; j < filterSpan; j++) {
                Neuron fromNeuron = neurons[i+j];
                Weight weight = Weights.randomWeight();
                filteredNeuron.getPredecessors().put(fromNeuron, weight);
                fromNeuron.getSuccessors().put(fromNeuron, weight);
            }

            filteredMap.add(filteredNeuron);
        }
    }

    @Override
    public int size() {
        return filteredMap.size();
    }

    @Override
    public Neuron get(int x) {
        return filteredMap.get(x);
    }

    @Override
    public void set(Neuron neuron, int x) {
        filteredMap.set(x, neuron);
    }

    @Override
    public Neuron[] toArray() {
        return filteredMap.toArray(new Neuron[0]);
    }

    @Override
    public Iterator<Neuron> iterator() {
        return filteredMap.iterator();
    }
}
