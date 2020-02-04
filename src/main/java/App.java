import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.Arrays;

public class App {
    private static final double LEARNING_RATE = 0.0002;
    //private static final double L2 = 0.005;
    private static final double GRADIENT_THRESHOLD = 100.0;
    private static final IUpdater UPDATER = Adam.builder().learningRate(LEARNING_RATE).beta1(0.5).build();
    private static final IUpdater UPDATER_ZERO = Sgd.builder().learningRate(0.0).build();

    private static JFrame frame;
    private static JPanel panel;

    private static double percent = 0.2d;
    private static int numbImages = 3;
    private static int width = (int)(percent*176);
    private static int height =(int)(percent*144);

    private static int seed = 10;
    private static int finalLayer = 1024;


    private static Layer[] genLayers() {
        return new Layer[] {
                new DenseLayer.Builder().nIn(64).nOut(256).weightInit(WeightInit.RELU).build(),
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new DenseLayer.Builder().nIn(256).nOut(512).build(),
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new DenseLayer.Builder().nIn(512).nOut(1024).build(),
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new DenseLayer.Builder().nIn(finalLayer).nOut(width*height*3).activation(Activation.TANH).build()
        };
    }

    /**
     * Returns a network config that takes in a 10x10 random number and produces a 28x28 grayscale image.
     *
     * @return config
     */
    private static MultiLayerConfiguration generator() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(UPDATER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(GRADIENT_THRESHOLD)
                //.l2(L2)
                .weightInit(WeightInit.RELU)
                .activation(Activation.IDENTITY)
                .list(genLayers())
                .build();

        return conf;
    }

    private static Layer[] disLayers(IUpdater updater) {
        return new Layer[] {
                new DenseLayer.Builder().nIn(width*height*3).nOut(finalLayer).updater(updater).build(),
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new DropoutLayer.Builder(1 - 0.5).build(),
                new DenseLayer.Builder().nIn(1024).nOut(512).updater(updater).build(),
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new DropoutLayer.Builder(1 - 0.5).build(),
                new DenseLayer.Builder().nIn(512).nOut(256).updater(updater).build(),
                new ActivationLayer.Builder(new ActivationLReLU(0.2)).build(),
                new DropoutLayer.Builder(1 - 0.5).build(),
                new OutputLayer.Builder(LossFunctions.LossFunction.XENT).nIn(256).nOut(1).activation(Activation.SIGMOID).updater(updater).build()
        };
    }

    private static MultiLayerConfiguration discriminator() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(UPDATER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(GRADIENT_THRESHOLD)
                //.l2(L2)
                .weightInit(WeightInit.RELU)
                .activation(Activation.IDENTITY)
                .list(disLayers(UPDATER))
                .build();

        return conf;
    }

    private static MultiLayerConfiguration gan() {
        Layer[] genLayers = genLayers();
        Layer[] disLayers = disLayers(UPDATER_ZERO); // Freeze discriminator layers in combined network.
        Layer[] layers = ArrayUtils.addAll(genLayers, disLayers);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(UPDATER)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .gradientNormalizationThreshold(GRADIENT_THRESHOLD)
                //.l2(L2)
                .weightInit(WeightInit.RELU)
                .activation(Activation.IDENTITY)
                .list(layers)
                .build();

        return conf;
    }

    public static void main(String... args) throws Exception {
        Nd4j.getMemoryManager().setAutoGcWindow(15 * 1000);

        float[][] data = DataSetter.getData(width,height);

        MultiLayerNetwork gen = new MultiLayerNetwork(generator());
        MultiLayerNetwork dis = new MultiLayerNetwork(discriminator());
        MultiLayerNetwork gan = new MultiLayerNetwork(gan());
        gen.init();
        dis.init();
        gan.init();

        copyParams(gen, dis, gan);

        gen.setListeners(new PerformanceListener(10, true));
        dis.setListeners(new PerformanceListener(10, true));
        gan.setListeners(new PerformanceListener(10, true));


        int j = 0;
        while (true) {

                //record how many generations have passed
                j++;
                if (j%5 == 0)
                System.out.println(j);

                // generate data
                INDArray real = new NDArray(data);
                int batchSize = (int)real.shape()[0];

                INDArray fakeIn = Nd4j.rand(new int[]{batchSize,  64});
                INDArray fake = gan.activateSelectedLayers(0, gen.getLayers().length - 1, fakeIn);

                DataSet realSet = new DataSet(real, Nd4j.zeros(batchSize, 1));
                DataSet fakeSet = new DataSet(fake, Nd4j.ones(batchSize, 1));

                DataSet newData = DataSet.merge(Arrays.asList(realSet, fakeSet));

                dis.fit(newData);
         //       dis.fit(newData);
                //dis.fit(realSet);
                //dis.fit(fakeSet);

                // Update the discriminator in the GAN network
                updateGan(gen, dis, gan);

                gan.fit(new DataSet(Nd4j.rand(new int[] { batchSize, 64}), Nd4j.zeros(batchSize, 1)));
                //gan.fit(fakeSet2);



                if (j % 10 == 1) {
                    System.out.println("Iteration " + j + " Visualizing...");
                    INDArray[] samples = new INDArray[numbImages];
                    DataSet fakeSet2 = new DataSet(fakeIn, Nd4j.ones(batchSize, 1));

                    for (int k = 0; k < numbImages; k++) {
                        INDArray input = fakeSet2.get(k).getFeatures();
                        //samples[k] = gen.output(input, false);
                        samples[k] = gan.activateSelectedLayers(0, gen.getLayers().length - 1, input);
                       // samples[k] = new NDArray(data[k]);

                    }
                    visualize(samples);
                }

        }
    }

    //copies the random setups of the gen and dis to the gan network
    private static void copyParams(MultiLayerNetwork gen, MultiLayerNetwork dis, MultiLayerNetwork gan) {
        int genLayerCount = gen.getLayers().length;
        for (int i = 0; i < gan.getLayers().length; i++) {
            if (i < genLayerCount) {
                gen.getLayer(i).setParams(gan.getLayer(i).params());
            } else {
                dis.getLayer(i - genLayerCount).setParams(gan.getLayer(i).params());
            }
        }
    }


    private static void updateGan(MultiLayerNetwork gen, MultiLayerNetwork dis, MultiLayerNetwork gan) {
        int genLayerCount = gen.getLayers().length;
        for (int i = genLayerCount; i < gan.getLayers().length; i++) {
            gan.getLayer(i).setParams(dis.getLayer(i - genLayerCount).params());
        }
    }

    private static void visualize(INDArray[] samples) {
        if (frame == null) {
            frame = new JFrame();
            frame.setTitle("Visualiser");
            frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
            frame.setLayout(new BorderLayout());

            panel = new JPanel();

            panel.setLayout(new GridLayout(samples.length / 3, 1, 8, 8));
            frame.add(panel, BorderLayout.CENTER);
            frame.setVisible(true);
        }

        panel.removeAll();

        for (int i = 0; i < samples.length; i++) {
            panel.add(getImage(samples[i]));
        }

        frame.revalidate();
        frame.pack();
    }

    //gets an image (RGB)
    private static JLabel getImage(INDArray tensor) {
        BufferedImage bi = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < height*width;i++){

            int x = i%width;
            int y = i/width;

                int red = Math.abs((int) (255*tensor.getFloat(i)));
                int green = Math.abs((int) (255*tensor.getFloat(i+width*height)));
                int blue = Math.abs((int) (255*tensor.getFloat(i+width*height*2)));
                int pixel = new Color(red,green,blue).getRGB();

                bi.setRGB(x,y,pixel);
        }

        ImageIcon orig = new ImageIcon(bi);
        Image imageScaled = orig.getImage().getScaledInstance((8 * width), (8 * height), Image.SCALE_REPLICATE);

        ImageIcon scaled = new ImageIcon(imageScaled);

        return new JLabel(scaled);
    }
}
