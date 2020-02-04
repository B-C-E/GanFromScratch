import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;

public class DataSetter
{

    public static float[][] getData(int width, int height)
    {
        try
        {
            File folder = new File(System.getProperty("user.home") + "/Downloads/New/");
            File[] listOfFiles = folder.listFiles();

            float[][]dataSet = new float[listOfFiles.length][width*height*3];
            int spot = -1;
            for (File image : listOfFiles)
            {
                spot++;
                float[] imageFloats = new float[width*height*3];

                BufferedImage img = ImageIO.read(image);
                BufferedImage resized = new BufferedImage(width, height, img.getType());
                Graphics2D g = resized.createGraphics();
                g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                        RenderingHints.VALUE_INTERPOLATION_BILINEAR);
                g.drawImage(img, 0, 0, width, height, 0, 0, img.getWidth(),
                        img.getHeight(), null);
                g.dispose();

                img = resized;



                for (int i = 0; i < height*width;i++){

                    int x = i%width;
                    int y = i/width;
                        Color atPixel = new Color(img.getRGB(x,y));


                        imageFloats[i] = atPixel.getRed()/255f;
                        imageFloats[i+width*height] = atPixel.getGreen()/255f;
                        imageFloats[i+width*height*2] = atPixel.getBlue()/255f;

                }
                dataSet[spot] = imageFloats;
            }

        return dataSet;
        }
        catch(Exception e)
        {}

        return null;
    }


}
