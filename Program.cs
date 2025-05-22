using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

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
        return Neurons.Select(n => n.Activate(inputs)).ToArray();
    }
}

public class NeuralNetwork
{
    public List<Layer> Layers = new();
    public double LearningRate = 0.1;

    public NeuralNetwork(int[] layerSizes, Random rand)
    {
        for (int i = 1; i < layerSizes.Length; i++)
            Layers.Add(new Layer(layerSizes[i], layerSizes[i - 1], rand));
    }

    public double[] FeedForward(double[] inputs)
    {
        foreach (var layer in Layers)
            inputs = layer.FeedForward(inputs);
        return inputs;
    }

    public void Train(double[] inputs, double[] targets)
    {
        var outputs = FeedForward(inputs);

        var outputLayer = Layers[^1];
        for (int i = 0; i < outputLayer.Neurons.Length; i++)
        {
            double error = targets[i] - outputLayer.Neurons[i].Output;
            outputLayer.Neurons[i].Delta = error * outputLayer.Neurons[i].SigmoidDerivative();
        }

        for (int l = Layers.Count - 2; l >= 0; l--)
        {
            var current = Layers[l];
            var next = Layers[l + 1];
            for (int i = 0; i < current.Neurons.Length; i++)
            {
                double sum = 0;
                for (int j = 0; j < next.Neurons.Length; j++)
                    sum += next.Neurons[j].Weights[i] * next.Neurons[j].Delta;
                current.Neurons[i].Delta = sum * current.Neurons[i].SigmoidDerivative();
            }
        }

        double[] prevOutputs = inputs;
        for (int l = 0; l < Layers.Count; l++)
        {
            if (l > 0)
                prevOutputs = Layers[l - 1].Neurons.Select(n => n.Output).ToArray();

            foreach (var neuron in Layers[l].Neurons)
            {
                for (int w = 0; w < neuron.Weights.Length; w++)
                    neuron.Weights[w] += LearningRate * neuron.Delta * prevOutputs[w];

                neuron.Bias += LearningRate * neuron.Delta;
            }
        }
    }

    public double[] Predict(double[] inputs) => FeedForward(inputs);

    public void SaveWeights(string filePath)
    {
        using var sw = new StreamWriter(filePath);
        foreach (var layer in Layers)
        {
            foreach (var neuron in layer.Neurons)
                sw.WriteLine(string.Join(' ', neuron.Weights.Select(w => w.ToString("R"))) + " B " + neuron.Bias.ToString("R"));
            sw.WriteLine("---");
        }
    }

    public void LoadWeights(string filePath)
    {
        var lines = File.ReadAllLines(filePath);
        int layerIndex = 0, neuronIndex = 0;

        foreach (var line in lines)
        {
            if (line == "---")
            {
                layerIndex++;
                neuronIndex = 0;
                continue;
            }

            var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            int split = Array.IndexOf(parts, "B");
            var weights = parts[..split].Select(double.Parse).ToArray();
            var bias = double.Parse(parts[^1]);

            var neuron = Layers[layerIndex].Neurons[neuronIndex];
            for (int i = 0; i < weights.Length; i++)
                neuron.Weights[i] = weights[i];
            neuron.Bias = bias;

            neuronIndex++;
        }
    }
}

public class CLI
{
    private NeuralNetwork network;
    private double[][] inputs;
    private double[][] targets;
    private bool dataLoaded = false;
    private bool networkInitialized = false;
    private Random rand = new();

    public void Run()
    {
        while (true)
        {
            Console.WriteLine("\n=== MENU ===");
            Console.WriteLine("1. Stwórz nową sieć neuronową");
            Console.WriteLine("2. Wczytaj dane z pliku");
            Console.WriteLine("3. Wczytaj wagi z pliku");
            Console.WriteLine("4. Zapisz wagi do pliku");
            Console.WriteLine("5. Trenuj sieć");
            Console.WriteLine("6. Pokaż predykcję dla wczytanych danych");
            Console.WriteLine("7. Pokaż status");
            Console.WriteLine("8. Sprawdź wynik dla własnych danych wejściowych");
            Console.Write("Wybierz opcję: ");

            string choice = Console.ReadLine();

            switch (choice)
            {
                case "1":
                    Console.Write("Podaj strukturę sieci (np. 3,3,2): ");
                    Init(Console.ReadLine());
                    break;
                case "2":
                    Console.Write("Podaj nazwę pliku z danymi (np. dane.txt): ");
                    LoadData(Console.ReadLine());
                    break;
                case "3":
                    Console.Write("Podaj nazwę pliku z wagami (np. wagi.txt): ");
                    LoadWeights(Console.ReadLine());
                    break;
                case "4":
                    Console.Write("Podaj nazwę pliku do zapisu wag (np. wagi.txt): ");
                    SaveWeights(Console.ReadLine());
                    break;
                case "5":
                    Console.Write("Podaj liczbę epok: ");
                    Train(Console.ReadLine());
                    break;
                case "6":
                    Predict();
                    break;
                case "7":
                    Status();
                    break;
                case "8":
                    ManualInput();
                    break;
                default:
                    Console.WriteLine("Nieprawidłowy wybór.");
                    break;
            }
        }
    }

    private void Init(string arg)
    {
        if (string.IsNullOrWhiteSpace(arg))
        {
            Console.WriteLine("Użycie: np. 3,4,2");
            return;
        }

        var parts = arg.Split(',').Select(int.Parse).ToArray();
        network = new NeuralNetwork(parts, rand);
        networkInitialized = true;
        Console.WriteLine("Sieć utworzona.");
    }

    private void LoadData(string fileName)
    {
        if (!networkInitialized)
        {
            Console.WriteLine("Najpierw stwórz sieć.");
            return;
        }

        if (!File.Exists(fileName))
        {
            Console.WriteLine("Plik nie istnieje.");
            return;
        }

        var lines = File.ReadAllLines(fileName);
        int inputSize = network.Layers[0].Neurons[0].Weights.Length;
        int outputSize = network.Layers[^1].Neurons.Length;

        inputs = new double[lines.Length][];
        targets = new double[lines.Length][];

        for (int i = 0; i < lines.Length; i++)
        {
            var values = lines[i].Split(' ').Select(double.Parse).ToArray();
            inputs[i] = values.Take(inputSize).ToArray();
            targets[i] = values.Skip(inputSize).Take(outputSize).ToArray();
        }

        dataLoaded = true;
        Console.WriteLine("Dane wczytane.");
    }

    private void Train(string arg)
    {
        if (!networkInitialized || !dataLoaded)
        {
            Console.WriteLine("Musisz utworzyć sieć i wczytać dane.");
            return;
        }

        if (!int.TryParse(arg, out int epochs))
        {
            Console.WriteLine("Podaj poprawną liczbę epok.");
            return;
        }

        for (int epoch = 1; epoch <= epochs; epoch++)
        {
            double totalError = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                network.Train(inputs[i], targets[i]);
                var output = network.Predict(inputs[i]);
                for (int j = 0; j < output.Length; j++)
                    totalError += Math.Pow(targets[i][j] - output[j], 2);
            }
            if (epoch % 100 == 0 || epoch == epochs)
                Console.WriteLine($"Epoka {epoch}, błąd: {Math.Round(totalError, 6)}");
        }
    }

    private void SaveWeights(string file)
    {
        if (!networkInitialized)
        {
            Console.WriteLine("Najpierw stwórz sieć.");
            return;
        }

        network.SaveWeights(file);
        Console.WriteLine("Wagi zapisane.");
    }

    private void LoadWeights(string file)
    {
        if (!networkInitialized)
        {
            Console.WriteLine("Najpierw stwórz sieć.");
            return;
        }

        network.LoadWeights(file);
        Console.WriteLine("Wagi wczytane.");
    }

    private void Predict()
    {
        if (!networkInitialized || !dataLoaded)
        {
            Console.WriteLine("Najpierw stwórz sieć i wczytaj dane.");
            return;
        }

        for (int i = 0; i < inputs.Length; i++)
        {
            var output = network.Predict(inputs[i]);
            Console.WriteLine($"{string.Join(' ', inputs[i])} → [{string.Join(", ", output.Select(o => Math.Round(o, 4)))}]");
        }
    }

    private void ManualInput()
    {
        if (!networkInitialized)
        {
            Console.WriteLine("Najpierw stwórz sieć.");
            return;
        }

        int inputSize = network.Layers[0].Neurons[0].Weights.Length;
        Console.WriteLine($"Podaj {inputSize} wartości oddzielonych spacją (np. 0 1 1):");
        Console.Write("> ");
        var line = Console.ReadLine();

        try
        {
            var input = line.Split(' ').Select(double.Parse).ToArray();
            if (input.Length != inputSize)
            {
                Console.WriteLine($"Błąd: Sieć oczekuje {inputSize} wejść.");
                return;
            }

            var output = network.Predict(input);
            Console.WriteLine($"Wynik sieci: [{string.Join(", ", output.Select(o => Math.Round(o, 4)))}]");
        }
        catch
        {
            Console.WriteLine("Błąd: Nie udało się przetworzyć danych wejściowych.");
        }
    }

    private void Status()
    {
        Console.WriteLine("\n=== STATUS ===");
        Console.WriteLine($"- Sieć utworzona: {(networkInitialized ? "Tak" : "Nie")}");
        Console.WriteLine($"- Dane wczytane: {(dataLoaded ? "Tak" : "Nie")}");
    }
}

public class Program
{
    public static void Main()
    {
        new CLI().Run();
    }
}
