package net.imglib2.trainable_segmention.ops;

import org.scijava.plugin.Plugin;

/**
 * @author Matthias Arzt
 */
@Plugin(type = FeatureOp.class, label = "Sobel Gradient Feature 2D")
public class SobelGradientFeature extends AbstractSigmaGroupFeatureOp {

	public SobelGradientFeature() {
		super(true);
	}

	@Override
	protected Class<? extends FeatureOp> getSingleFeatureClass() {
		return SingleSobelGradientFeature.class;
	}
}
