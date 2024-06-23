package examples;

import science.nn.functional.*;
import science.nn.layer.*;
import science.nn.loss.*;
import science.nn.model.*;
import science.nn.optim.*;

import java.util.Arrays;

public class NNExample {



    private static final int TRAINING_ITERATIONS = 1_000_000;


    public static void main(String[] args) {

        Sequence model = new Sequence();

        model.addLayer(new Dense(4, new Tanh(), 0));
        model.addLayer(new Dense(4, new Tanh(), 1));
        model.addLayer(new Dense(4, new Tanh(), 1));
        model.addLayer(new Dense(4, new Tanh(), 1));
        model.addLayer(new Dense(3, new Swish(), 1));

        double[] input = new double[]{0, 1, 1, 0};
        double[] target = new double[]{1, 0, 1};

        test(model, input, target);
        train(model, input, target, TRAINING_ITERATIONS);
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