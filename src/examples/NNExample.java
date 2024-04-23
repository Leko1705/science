package examples;

import science.nn.functional.Swish;
import science.nn.layer.Dense;
import science.nn.loss.Loss;
import science.nn.loss.MSELoss;
import science.nn.model.Model;
import science.nn.model.Sequence;
import science.nn.optim.SGD;

import java.util.Arrays;

public class NNExample {
    public static void main(String[] args) {

        Sequence model = new Sequence();

        model.addLayer(new Dense(4, new Swish(), 1.0));
        model.addLayer(new Dense(5, new Swish(), 1.0));
        model.addLayer(new Dense(5, new Swish(), 1.0));
        model.addLayer(new Dense(1, new Swish(), 1.0));

        double[] input = new double[]{0, 1, 1, 0};
        double[] target = new double[]{1};

        test(model, input, target);
        train(model, input, target, 100_000);
        test(model, input, target);
    }

    private static void test(Model model, double[] input, double[] target) {
        double[] output = model.generate(input);
        System.out.println("target:" + Arrays.toString(target));
        System.out.println("output: " + Arrays.toString(output));
        Loss loss = new MSELoss(output, target);
        System.out.println("loss: " + loss.calculate() + "\n");
    }

    private static void train(Model model, double[] input, double[] target, int durations) {
        for (int i = 0; i < durations; i++) {
            double[] output = model.generate(input);
            Loss loss = new MSELoss(output, target);
            model.zeroGradients();
            model.backward(loss);
            model.fit(new SGD());
        }
    }
}