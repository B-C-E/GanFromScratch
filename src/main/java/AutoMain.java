import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.activations.impl.ActivationTanH;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;

public class AutoMain
{

    //@Todo
    //Actual AutoEncoding network
    //Displayer
    //Saving, loading, etc

    ///////////////////////////////////////////////////////////////////////////////////////
    //Training Here: /////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    public static void main(String args[])
    {
        MultiLayerNetwork auto = new MultiLayerNetwork(autoEncoderConfig());
        auto.setListeners(new PerformanceListener(1, true));
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    //The Setup for our network goes here /////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    //Associated Variables
    private static int dimensions = 12;//how many dimensions are we compressing to?
    private static int seed = 735;
    private static final IUpdater UPDATER = AdaGrad.builder().learningRate(0.05).build();//AdaGrad will be updating our network for us, using momentum and the like


    //This is a method for visual clarity
    private static Layer[] autoEncoderLayers(IUpdater updater) {
        return new Layer[] {
                new DenseLayer.Builder().nIn(width*height*3).nOut((width*height)/2).updater(updater).build(),
                new DenseLayer.Builder().nIn((width*height)/2).nOut(dimensions).updater(updater).build(),
                new ActivationLayer.Builder(new ActivationTanH()).build(),//this should contrain the values to between -1 and 1
                new DenseLayer.Builder().nIn(dimensions).nOut((width*height)/2).updater(updater).build(),
                new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(width*height/2).nOut(width*height*3).activation(Activation.IDENTITY).updater(updater).build()

                //LossFunctions.LossFunction.MSE -- Uses the mean square loss function
                //Activation - IDENTITY -- Activations can be used to constrain data within bounds. IDENTITY does nothing

        };
    }


    private static MultiLayerConfiguration autoEncoderConfig()
    {
        return new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(UPDATER)//this is a static variable, see above
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .weightInit(WeightInit.RELU)//what kind of random does it start with?
                .activation(Activation.IDENTITY)//does nothing
                .list(autoEncoderLayers(UPDATER))
                .build();
    }



    ///////////////////////////////////////////////////////////////////////////////////////
    //Methods for Managing Data Here /////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    //Associated Variables:
    private static int width = 24;//input data will be scaled down to this width
    private static int height = 16;//by this height

    //Gets a float[][] array that will have all of your training data in it
    //
    //Takes in a DESIRED width and height (images will be scaled to this size)
    //And the path to where the training data is (where the images are)
    public static float[][] getData(int width, int height, File whereTheImagesAre)
    {
        try
        {

            //File folder = new File(System.getProperty("user.home") + "/Downloads/TrainingFiles/");

            //Gets all of the images as Files
            File[] listOfFiles = whereTheImagesAre.listFiles();

            //creates a nice, empty dataset with as many slots as there are files in our folder
            float[][]dataSet = new float[listOfFiles.length][width*height*3];


            //this loop reads each file and adds it to the dataset.
            //spot keeps track of where in [][]dataSet to add the latest image
            int spot = 0;
            for (File image : listOfFiles)
            {

                float[] imageFloats = new float[width*height*3];//creates an empty array for our latest image

                //get and resize the image
                BufferedImage img = ImageIO.read(image);
                BufferedImage resized = new BufferedImage(width, height, img.getType());
                Graphics2D g = resized.createGraphics();
                g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                        RenderingHints.VALUE_INTERPOLATION_BILINEAR);
                g.drawImage(img, 0, 0, width, height, 0, 0, img.getWidth(),
                        img.getHeight(), null);
                g.dispose();

                img = resized;
                //the image is now resized


                //get all the actual values from the image
                for (int i = 0; i < height*width;i++){

                    //figure out where from the image to get the color of
                    int x = i%width;
                    int y = i/width;
                    Color atPixel = new Color(img.getRGB(x,y));


                    //get the red green and blue values, add to our array
                    imageFloats[i] = atPixel.getRed()/255f;
                    imageFloats[i+width*height] = atPixel.getGreen()/255f;
                    imageFloats[i+width*height*2] = atPixel.getBlue()/255f;

                }


                dataSet[spot] = imageFloats;//insert converted image into the dataSet
                spot++;//increment the spot we are inserting things, so the next image goes in the next spot
            }//end of adding all of the images

            return dataSet;
        }
        catch(Exception e)
        {
            System.out.println("An error occured during Data Set generation");
            System.out.println(e.getStackTrace());
        }

        return null;//if data set generation fails, return null, which will cause things to break and shutdown.
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    //Methods for displaying stuff here: /////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    //Associated variables:
    private static JFrame frame;
    private static JPanel panel;
    private static int rows = 4; //How many rows of images/graphs/etc will be displayed?
    private static int columns = 4; //how many columns will be displayed?

    //Displays what's new with our gan. Takes in a collection of INDArrays to turn into images
    private static void display(INDArray[] images) {
        if (frame == null) //if no frame has been made yet
        {
            //make a new frame
            frame = new JFrame();
            frame.setTitle("Latest Images");
            frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
            frame.setLayout(new BorderLayout());

            panel = new JPanel();

            panel.setLayout(new GridLayout(rows,columns,0,0));
            frame.add(panel, BorderLayout.CENTER);
            frame.setVisible(true);
        }

        panel.removeAll();

        for (int i = 0; i < images.length; i++) {
            //turns the image array into a buffered image which is turned into an image icon
            //which is turned into a JLabel, which is then added to the panel.
            panel.add(new JLabel(new ImageIcon(getImageRGB(images[i],width,height))));
        }

        frame.revalidate();
        frame.pack();
    }//end of display

    //gets a RGB buffered image from an INDArray
    //
    //  The inputArray should be structured as follows:
    //  Individual values are the brightness of either red, green or blue
    //  they are calculated by taking, for example, the red value of a pixel, and dividing it by 255 (they are stored as floats, I believe)
    //  This way, if the pixel is all the way red, it will be at the max value of 255, and 255/255 = 1,
    //  if the pixel has no red in it, 0/255 is 0,
    //  and if it is in the middle, say 128, 128/255 == 0.501 ish.
    //
    //The input array must have all of the red values (calculated as above), then all of the greens, then all of the blues
    //
    //The width and height are the width and height of the image
    private static BufferedImage getImageRGB(INDArray inputArray, int width, int height) {
        //the image to return (B uffered I mage --> bi). Sorry, I guess that's pretty obvious.
        BufferedImage bi = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

        //go through the 1/3 of the INDArray (the array is width * height for red, green and blue, so just width
        // times height is 1/3 of it)
        for (int i = 0; i < width*height;i++)//This turns the array into our image
        {
            //calculate where the x and y values should be on the actual image
            int x = i%width;
            int y = i/width;

            //find the reds, greens, and blues
            int red = (int) (255*inputArray.getFloat(i)); //get the red directly from the spot
            int green = (int) (255*inputArray.getFloat(i+width*height)); //get the green from the spot 1/3 of the array forwards
            int blue = (int) (255*inputArray.getFloat(i+width*height*2)); //get the blue from the spot 2/3 of the array forwards

            //There is a possibility that the AI has generated impossible colors
            //(IE darker than black (RGB = -20, -5, 0, for example) or brighter than white
            //clamp these values to get something decent (clamping forces the values to be between the min and max)
            red = clamp(red,0,255);
            green = clamp(green,0,255);
            blue = clamp(blue,0,255);

            //get the color for the pixel (stored as an int)
            int colorOfPixel = new Color(red,green,blue).getRGB();

            //set the color at the spot
            bi.setRGB(x,y,colorOfPixel);

        }//end of coloring in our buffered image. It is now correctly made

        return bi;
    }//end of getImageRGB

    //Scales a buffered image
    private static BufferedImage scaleImage(BufferedImage image, double xScale, double yScale)
    {
        int width = image.getWidth();
        int height = image.getHeight();

        Image imageScaled =  image.getScaledInstance((int)(xScale * width), (int)(yScale * height), Image.SCALE_REPLICATE);

        return (BufferedImage) imageScaled;
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    //RANDOM UTILITY METHODS HERE:: /////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////

    //clamps value between min and max
    public static int clamp(int value, int min, int max)
    {
        return Math.max(min,Math.min(max,value));
    }


}//The end!