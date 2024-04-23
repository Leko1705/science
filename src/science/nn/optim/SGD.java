package science.nn.optim;

import science.nn.graph.Neuron;
import science.nn.graph.Weight;
import science.nn.layer.Layer;

import java.util.Map;


@SuppressWarnings("unused")
public class SGD implements Optimizer {

    private double lernrate;


    public SGD(){
        this.lernrate = 0.01;
    }

    public SGD(double lernrate){
        this.lernrate = lernrate;
    }

    @Override
    public void optimize(Layer layer) {

        for (Neuron neuron : layer.getShape()){
            double activationLevel = neuron.getValue();
            Map<Neuron, Weight> weights = neuron.getSuccessors();

            for (Neuron successor : weights.keySet()){
                double gradient = successor.getGradient();
                double delta = gradient * activationLevel;

                Weight w = weights.get(successor);
                w.set(w.get() + (lernrate * -delta));
            }
        }

    }

    public double getLernrate() {
        return lernrate;
    }

    public void setLernrate(double lernrate) {
        this.lernrate = lernrate;
    }
}
