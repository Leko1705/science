package science.nn.optim;

import science.nn.graph.Neuron;
import science.nn.graph.Weight;
import science.nn.layer.Layer;

import java.util.Map;

public class Adam implements Optimizer{

    private double lernrate = 0.01;
    private double alpha = 0.9;
    private double beta = 0.999;
    private double epsilon = Math.pow(10, -8);

    private double sdwPrev = 0;
    private double sdbPrev;

    private double vdwPrev = 0;
    private double vdbPrev;

    public Adam(){
    }

    public Adam(double lernrate){
        this.lernrate = lernrate;
    }

    public Adam(double lernrate, double beta){
        this.lernrate = lernrate;
        this.beta = beta;
    }

    public Adam(double lernrate, double beta, double epsilon){
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

                sdwPrev = alpha * sdwPrev + (1 - alpha) * Math.pow(delta, 2);
                vdwPrev = beta * vdwPrev + (1 - beta) * delta;

                delta = vdwPrev / Math.sqrt(sdwPrev + epsilon);

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

    public double getAlpha() {
        return alpha;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
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
