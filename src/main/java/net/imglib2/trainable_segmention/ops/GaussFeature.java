package net.imglib2.trainable_segmention.ops;

import org.scijava.plugin.Plugin;

/**
 * @author Matthias Arzt
 */
@Plugin(type = FeatureOp.class)
public class GaussFeature extends AbstractSigmaGroupFeatureOp {

	public GaussFeature() {
		super(false);
	}

	@Override
	protected Class<? extends FeatureOp> getSingleFeatureClass() {
		return SingleGaussFeature.class;
	}
}
