package science.recurrent;

import java.util.Arrays;

public class Hopfield {

    private final double[] neurons;
    private final double[][] wights;

    public Hopfield(int neuronAmount){
        neurons = new double[neuronAmount];
        wights = new double[neuronAmount][neuronAmount];
        for (int i = 0; i < neuronAmount; i++){
            neurons[i] = 0.0;
            Arrays.fill(wights[i], 0.0);
        }
    }

    public void store(double[] data){
        if (data.length != neurons.length)
            throw new IllegalArgumentException("input size does not match hopfield size");

        for (int i = 0; i < neurons.length; i++){
            for (int j = 0; j < neurons.length; j++){
                if (i != j) {
                    wights[i][j] += (data[i] * data[j])/neurons.length;
                }
            }
        }
    }

    public double[] load(double[] data){
        if (data.length != neurons.length)
            throw new IllegalArgumentException("input size does not match hopfield size");

        for (int x = 0; x < neurons.length; x++){
            double sum = 0.0;
            for (int i = 0; i < data.length; i++)
            {
                sum += wights[x][i] * data[i];
            }
            neurons[x] = sum >= 0 ? 1 : -1;
        }
        return neurons;
    }

    public double[][] getWights() {
        return wights;
    }

}
