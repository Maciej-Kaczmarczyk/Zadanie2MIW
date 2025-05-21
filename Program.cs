using System;
using System.Collections.Generic;

public class Neuron
{
    public double[] Weights;
    public double Bias;
    public double Output;
    public double Delta;

    public Neuron(int inputCount, Random rand)
    {
        Weights = new double[inputCount];
        for (int i = 0; i < inputCount; i++)
            Weights[i] = rand.NextDouble() * 2 - 1;
        Bias = rand.NextDouble() * 2 - 1;
    }

    public double Activate(double[] inputs)
    {
        double sum = Bias;
        for (int i = 0; i < inputs.Length; i++)
            sum += inputs[i] * Weights[i];
        Output = 1.0 / (1.0 + Math.Exp(-sum));
        return Output;
    }

    public double SigmoidDerivative()
    {
        return Output * (1 - Output);
    }
}

public class Layer
{
    public Neuron[] Neurons;

    public Layer(int neuronCount, int inputCount, Random rand)
    {
        Neurons = new Neuron[neuronCount];
        for (int i = 0; i < neuronCount; i++)
            Neurons[i] = new Neuron(inputCount, rand);
    }

    public double[] FeedForward(double[] inputs)
    {
        double[] outputs = new double[Neurons.Length];
        for (int i = 0; i < Neurons.Length; i++)
            outputs[i] = Neurons[i].Activate(inputs);
        return outputs;
    }
}

public class NeuralNetwork
{
    private List<Layer> Layers = new();
    private double LearningRate = 0.1;

    public NeuralNetwork(int[] layerSizes, Random rand)
    {
        for (int i = 1; i < layerSizes.Length; i++)
        {
            int inputCount = layerSizes[i - 1];
            int neuronCount = layerSizes[i];
            Layers.Add(new Layer(neuronCount, inputCount, rand));
        }
    }
    public double[] FeedForward(double[] inputs)
    {
        double[] outputs = inputs;
        foreach (var layer in Layers)
            outputs = layer.FeedForward(outputs);
        return outputs;
    }

    public void Train(double[] inputs, double[] targets)
    {
        double[] outputs = FeedForward(inputs);

        Layer outputLayer = Layers[^1];
        for (int i = 0; i < outputLayer.Neurons.Length; i++)
        {
            double error = targets[i] - outputLayer.Neurons[i].Output;
            outputLayer.Neurons[i].Delta = error * outputLayer.Neurons[i].SigmoidDerivative();
        }

        for (int l = Layers.Count - 2; l >= 0; l--)
        {
            Layer current = Layers[l];
            Layer next = Layers[l + 1];
            for (int i = 0; i < current.Neurons.Length; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < next.Neurons.Length; j++)
                    sum += next.Neurons[j].Weights[i] * next.Neurons[j].Delta;

                current.Neurons[i].Delta = sum * current.Neurons[i].SigmoidDerivative();
            }
        }

        double[] prevOutputs = inputs;
        for (int l = 0; l < Layers.Count; l++)
        {
            Layer layer = Layers[l];

            if (l > 0)
            {
                prevOutputs = new double[Layers[l - 1].Neurons.Length];
                for (int n = 0; n < Layers[l - 1].Neurons.Length; n++)
                    prevOutputs[n] = Layers[l - 1].Neurons[n].Output;
            }

            foreach (var neuron in layer.Neurons)
            {
                for (int w = 0; w < neuron.Weights.Length; w++)
                    neuron.Weights[w] += LearningRate * neuron.Delta * prevOutputs[w];

                neuron.Bias += LearningRate * neuron.Delta;
            }
        }
    }

    public double[] Predict(double[] inputs)
    {
        return FeedForward(inputs);
    }
}

public class Program
{
    public static void Main()
    {
        var rand = new Random();

        int[] structure = { 3, 3, 2, 2 };
        var nn = new NeuralNetwork(structure, rand);

        double[][] inputs = {
            new double[] { 0, 0, 0 },
            new double[] { 0, 1, 0 },
            new double[] { 1, 0, 0 },
            new double[] { 1, 1, 0 },
            new double[] { 0, 0, 1 },
            new double[] { 0, 1, 1 },
            new double[] { 1, 0, 1 },
            new double[] { 1, 1, 1 }
        };

        double[][] targets = {
            new double[] { 0, 0 },
            new double[] { 1, 0 },
            new double[] { 1, 0 },
            new double[] { 0, 1 },
            new double[] { 1, 0 },
            new double[] { 0, 1 },
            new double[] { 0, 1 },
            new double[] { 1, 1 }
        };

        for (int epoch = 0; epoch < 50000; epoch++)
        {
            double totalError = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                nn.Train(inputs[i], targets[i]);
                var output = nn.Predict(inputs[i]);
                for (int j = 0; j < output.Length; j++)
                    totalError += Math.Pow(targets[i][j] - output[j], 2);
            }
            Console.WriteLine($"Epoka {epoch + 1}, błąd: {Math.Round(totalError, 6)}");
        }

        Console.WriteLine("\nPredykcja:");
        for (int i = 0; i < inputs.Length; i++)
        {
            var output = nn.Predict(inputs[i]);
            Console.WriteLine($"{inputs[i][0]} {inputs[i][1]} {inputs[i][2]} → [{Math.Round(output[0], 4)}, {Math.Round(output[1], 4)}]");
        }
    }
}