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

public class Conv2D implements Layer2D, Shape {

    private static final Function IDENTITY = new Identity();


    private final Weight[][] filter;

    private final List<Neuron> filteredMap = new ArrayList<>();

    private int span;


    public Conv2D(int filterSpan) {
        this.filter = new Weight[filterSpan][filterSpan];
    }


    @Override
    public Kind getKind() {
        return Kind.FILTERED;
    }

    @Override
    public void forward() {
        for (Neuron filtered : filteredMap) {

            double sum = 0;
            for (Neuron pred : filtered.getPredecessors().keySet()) {
                Weight weight = filtered.getPredecessors().get(pred);
                sum += pred.getValue() * weight.get();
            }

            filtered.setValue(sum / Math.pow(span, 2));
        }
    }

    @Override
    public void backward() {
        // TODO impl backward pass for this layer
    }

    @Override
    public Shape getShape() {
        return this;
    }

    @Override
    public void connectPrevious(Layer layer) {

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

        span = inputDim-filter.length+1;

        for (int i = 0; i < span; i++) {
            for (int j = 0; j < span; j++) {
                Neuron filteredNeuron = new SimpleNeuron(IDENTITY);

                for (int k = 0; k < filter.length; k++) {
                   for (int l = 0; l < filter.length; l++) {
                       Neuron fromNeuron = neurons[i+k][j+l];
                       Weight weight = Weights.randomWeight();

                       filteredNeuron.getPredecessors().put(fromNeuron, weight);
                       fromNeuron.getSuccessors().put(fromNeuron, weight);
                       filter[k][l] = weight;
                   }
                }

                filteredMap.add(filteredNeuron);
            }
        }
    }

    @Override
    public int getSpan() {
        return span;
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
