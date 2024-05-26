package science.nn.layer;

import science.nn.graph.Neuron;

public interface Shape extends Iterable<Neuron> {

    default int size() {
        int size = 0;
        for (Neuron ignored : this) {
            size++;
        }
        return size;
    }

    Neuron get(int x);

    void set(Neuron neuron, int x);

    default Neuron[] toArray(){
        Neuron[] array = new Neuron[size()];
        int i = 0;
        for (Neuron neuron : this) {
            array[i] = neuron;
        }
        return array;
    }

}
