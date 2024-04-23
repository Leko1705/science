package science.nn.graph;

public class SimpleWeight implements Weight {

    private double value;

    public SimpleWeight(double value) {
        this.value = value;
    }

    @Override
    public double get() {
        return value;
    }

    @Override
    public void set(double value) {
        this.value = value;
    }
}
