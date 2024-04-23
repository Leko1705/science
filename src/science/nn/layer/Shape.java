package science.nn.layer;

import science.nn.graph.Neuron;

public interface Shape extends Iterable<Neuron> {

    int[] getDimensions();

    default int getRank(){
        return getDimensions().length;
    }

    Neuron get(int x, int... y);

    void set(Neuron neuron, int x, int... y);

}
