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

public class PollingLayer2D implements Layer2D, Shape {

    private static final Function IDENTITY = new Identity();

    private final List<Neuron> filteredMap = new ArrayList<>();

    private final int filterSpan;

    private int span;

    private final Poller2D poller2D;

    public PollingLayer2D(Poller2D poller2D, int filterSpan) {
        this.poller2D = poller2D;
        this.filterSpan = filterSpan;
    }

    @Override
    public int getSpan() {
        return span;
    }

    @Override
    public Kind getKind() {
        return Kind.FILTERED;
    }

    @Override
    public void forward() {
        for (Neuron neuron : filteredMap) {
            Neuron[][] batch = new Neuron[filterSpan][filterSpan];

            Iterator<Neuron> iterator = neuron.getPredecessors().keySet().iterator();
            for (int i = 0; i < filterSpan; i++) {
                for (int j = 0; j < filterSpan; j++) {
                    Neuron n = iterator.next();
                    batch[i][j] = n;
                }
            }

            double polled = poller2D.poll(batch);
            neuron.setValue(polled);
        }
    }

    @Override
    public void backward() {
        for (Neuron neuron : filteredMap) {
            neuron.setGradient(0);
        }
    }

    @Override
    public Shape getShape() {
        return this;
    }

    @Override
    public void connectPrevious(Layer layer) {
        if (layer.getKind() != Kind.FILTERED)
            throw new IncompatibleLayerException("Polling layer requires a filtered layer as a predecessor");

        int size = layer.getShape().size();
        int inputDim = (int) Math.sqrt(size);

        Neuron[][] neurons = new Neuron[inputDim][inputDim];
        int x = 0, y = 0;
        for (Neuron neuron : layer.getShape()){
            neurons[x][y] = neuron;
            x++;
            if (x == inputDim) {
                x = 0;
                y++;
            }
        }

        span = inputDim-filterSpan+1;

        for (int i = 0; i < span; i++) {
            for (int j = 0; j < span; j++) {
                Neuron filteredNeuron = new SimpleNeuron(IDENTITY);

                for (int k = 0; k < filterSpan; k++) {
                    for (int l = 0; l < filterSpan; l++) {
                        Neuron fromNeuron = neurons[i+k][j+l];
                        Weight weight = Weights.randomWeight();
                        filteredNeuron.getPredecessors().put(fromNeuron, weight);
                        fromNeuron.getSuccessors().put(fromNeuron, weight);
                    }
                }

                filteredMap.add(filteredNeuron);
            }
        }
    }

    @Override
    public int size() {
        return filteredMap.size();
    }

    @Override
    public Neuron[] toArray() {
        return filteredMap.toArray(new Neuron[0]);
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
    public Iterator<Neuron> iterator() {
        return filteredMap.iterator();
    }
}
