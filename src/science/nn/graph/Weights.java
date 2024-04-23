package science.nn.graph;

public class Weights {

    public static Weight randomWeight() {
        double r = Math.random();
        if (Math.random() < 0.4) r *= -1;
        return new SimpleWeight(r);
    }

    public static Weight of(double weight) {
        return new SimpleWeight(weight);
    }

}
