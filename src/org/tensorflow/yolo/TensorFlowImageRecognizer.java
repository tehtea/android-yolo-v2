package org.tensorflow.yolo;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

import org.tensorflow.yolo.model.PostProcessingOutcome;
import org.tensorflow.yolo.util.ClassAttrProvider;

import me.daquexian.dabnn.Net;

import java.io.IOException;
import java.util.Vector;

import static org.tensorflow.yolo.Config.IMAGE_MEAN;
import static org.tensorflow.yolo.Config.IMAGE_STD;
import static org.tensorflow.yolo.Config.INPUT_NAME;
import static org.tensorflow.yolo.Config.INPUT_SIZE;
import static org.tensorflow.yolo.Config.MODEL_FILE;
import static org.tensorflow.yolo.Config.OUTPUT_CHANNELS;
import static org.tensorflow.yolo.Config.OUTPUT_NAME;
import static org.tensorflow.yolo.Config.OUTPUT_WIDTH;

/**
 * A classifier specialized to label images using TensorFlow.
 * Modified by Zoltan Szabo
 */
public class TensorFlowImageRecognizer {
    private int outputSize;
    private Vector<String> labels;
    private Net inferenceInterface;

    private TensorFlowImageRecognizer() {
    }

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager The asset manager to be used to load assets.
     * @throws IOException
     */
    public static TensorFlowImageRecognizer create(AssetManager assetManager) {
        TensorFlowImageRecognizer recognizer = new TensorFlowImageRecognizer();
        recognizer.labels = ClassAttrProvider.newInstance(assetManager).getLabels();
        recognizer.inferenceInterface = new Net().readAsset(assetManager, MODEL_FILE);
        recognizer.outputSize = (int) (OUTPUT_CHANNELS * Math.pow(OUTPUT_WIDTH, 2)); // TODO: find out how to get this
        return recognizer;
    }

    public PostProcessingOutcome recognizeImage(final Bitmap bitmap) {
        return YOLOClassifier.getInstance().classifyImage(runTensorFlow(bitmap), labels);
    }

    public String getStatString() {
        return "Hi"; // TODO: find out what to put for this
    }

    public void close() {
        inferenceInterface.dispose();
    }

    private float[] runTensorFlow(final Bitmap bitmap) {

        inferenceInterface.predict(processBitmap(bitmap));

        return inferenceInterface.getBlob(OUTPUT_NAME);
    }

    /**
     * Preprocess the image data from 0-255 int to normalized float based
     * on the provided parameters.
     *
     * @param bitmap
     */
    private float[] processBitmap(final Bitmap bitmap) {
        int[] intValues = new int[INPUT_SIZE * INPUT_SIZE];
        float[] floatValues = new float[INPUT_SIZE * INPUT_SIZE * 3];

        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3 + 0] = (((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
            floatValues[i * 3 + 1] = (((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
            floatValues[i * 3 + 2] = ((val & 0xFF) - IMAGE_MEAN) / IMAGE_STD;
        }
        return floatValues;
    }
}
