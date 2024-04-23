package science.nn.layer;

import science.nn.graph.Neuron;

public class DropoutLayer implements Layer {

    private Layer prevLayer;

    private final int durations;
    private final double probability;

    private int iterations = 0;

    public DropoutLayer(int durations, double probability) {
        checkInput(durations, probability);
        this.durations = durations;
        this.probability = probability;
    }

    private void checkInput(int durations, double probability) {
        if (durations <= 0)
            throw new IllegalArgumentException("durations must be > 0");
        if (probability < 0 || probability > 1)
            throw new IllegalArgumentException("probability must be between 0 and 1");
    }

    @Override
    public Kind getKind() {
        return prevLayer.getKind();
    }

    @Override
    public void forward() {
        iterations++;
        if (iterations % durations == 0) {
            for (Neuron neuron : getShape()) {
                if (Math.random() < probability) {
                    neuron.setValue(0);
                }
            }
        }
    }

    @Override
    public void backward() {

    }

    @Override
    public Shape getShape() {
        return prevLayer.getShape();
    }

    @Override
    public void connectPrevious(Layer layer) {
        this.prevLayer = layer;
    }
}
