package science.nn.graph;

import science.nn.functional.Function;

import java.util.HashMap;
import java.util.Map;

public class SimpleNeuron implements Neuron {

    private double value;

    private double gradient;

    private Function function;

    private double unsquashed;

    private final Map<Neuron, Weight> predecessors = new HashMap<>();

    private final Map<Neuron, Weight> successors = new HashMap<>();

    public SimpleNeuron(Function function) {
        this.function = function;
    }

    public Function getFunction() {
        return function;
    }

    public void setFunction(Function function) {
        this.function = function;
    }

    @Override
    public double getValue() {
        return value;
    }

    @Override
    public void setValue(double value) {
        this.value = value;
    }

    @Override
    public void setGradient(double gradient) {
        this.gradient = gradient;
    }

    @Override
    public double getGradient() {
        return gradient;
    }

    @Override
    public Map<Neuron, Weight> getPredecessors() {
        return predecessors;
    }

    @Override
    public Map<Neuron, Weight> getSuccessors() {
        return successors;
    }

    @Override
    public double squash(double x) {
        unsquashed = x;
        return function.eval(x);
    }

    @Override
    public double derivative(double x) {
        return function.gradient(x);
    }

    @Override
    public double getUnsquashed() {
        return unsquashed;
    }
}
