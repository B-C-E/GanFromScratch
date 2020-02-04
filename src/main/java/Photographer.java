import com.github.sarxos.webcam.Webcam;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;

public class Photographer
{

    public static void main(String args[])
    {
        try
        {
            Webcam webcam = Webcam.getWebcams().get(0);
            webcam.open();
            int i = 0;
            int numbOfImages = 250;
            String downloads = System.getProperty("user.home") + "/Downloads/New/";

            while (i<numbOfImages)
            {
                Thread.sleep(3);
                i++;
                System.out.println("Got one. (" + i +")");
                BufferedImage image = webcam.getImage();
                ImageIO.write(image, "JPG", new File(downloads + i+".jpg"));

            }
        }catch(Exception e)
        {
            System.out.println("Uh oh.");
        }
    }
}

