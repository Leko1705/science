package examples;

import science.recurrent.Hopfield;

import java.util.Arrays;

public class HopfieldExample {


    private static final double[][] DATA = {
            {0, 0, -1, 1, 0},
            {1, 0, 0, -1, 0},
    };

    private static final double[][] DEVIATIONS = {
            {1, 0, -1, 1, 0}, // first 0 changed to 1
            {1, 0, 0, -1, 1}, // last 0 changed to 1
    };


    public static void main(String[] args) {
        Hopfield hf = new Hopfield(5);

        for (double[] set : DATA) {
            hf.store(set);
        }

        for (int i = 0; i < DATA.length; i++) {
            double[] res = hf.load(DEVIATIONS[i]);
            double[] target = DATA[i];

            System.out.println("target: " + Arrays.toString(target));
            System.out.println("output: " + Arrays.toString(res) + " (deviation: " + Arrays.toString(DEVIATIONS[i]) + ")");
        }
    }

}
