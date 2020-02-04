import org.nd4j.linalg.api.ndarray.INDArray;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

public class GanMain
{

    //@Todo
    //Actual GAN networks
    //Displayer


    ///////////////////////////////////////////////////////////////////////////////////////
    //Methods for displaying stuff here: /////////////////////////////////////////////////////
    ///////////////////////////////////////////////////////////////////////////


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

    //RANDOM UTILITY METHODS HERE:

    //clamps value between min and max
    public static int clamp(int value, int min, int max)
    {
        return Math.max(min,Math.min(max,value));
    }
}
