package science.nn.optim;

import science.nn.graph.Neuron;
import science.nn.graph.Weight;
import science.nn.layer.Layer;

import java.util.Map;

public class RMSprop implements Optimizer{

    private double lernrate = 0.01;
    private double beta = 0.999;
    private double epsilon = Math.pow(10, -8);

    private double sdwPrev = 0;
    private double sdbPrev;

    public RMSprop(){

    }

    public RMSprop(double lernrate){
        this.lernrate = lernrate;
    }

    public RMSprop(double lernrate, double beta){
        this.lernrate = lernrate;
        this.beta = beta;
    }

    public RMSprop(double lernrate, double beta, double epsilon){
        this.lernrate = lernrate;
        this.beta = beta;
        this.epsilon = epsilon;
    }

    @Override
    public void optimize(Layer layer) {

        for (Neuron neuron : layer.getShape()){
            double activationLevel = neuron.getValue();
            Map<Neuron, Weight> weights = neuron.getSuccessors();

            for (Neuron successor : weights.keySet()){
                double gradient = successor.getGradient();
                double delta = gradient * activationLevel;

                sdwPrev = beta * sdwPrev + (1 - beta) * Math.pow(delta, 2);;
                delta /= Math.sqrt(sdwPrev + epsilon);

                Weight weight = weights.get(successor);
                weight.set(weight.get() + (lernrate * -delta));
            }
        }

    }

    public double getLernrate() {
        return lernrate;
    }

    public void setLernrate(double lernrate) {
        this.lernrate = lernrate;
    }

    public double getBeta() {
        return beta;
    }

    public void setBeta(double beta) {
        this.beta = beta;
    }

    public double getEpsilon() {
        return epsilon;
    }

    public void setEpsilon(double epsilon) {
        this.epsilon = epsilon;
    }
}
