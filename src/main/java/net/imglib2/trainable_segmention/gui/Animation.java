package net.imglib2.trainable_segmention.gui;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.Timer;

public abstract class Animation implements ActionListener {

	/** start value (typically in pixels) */
	private int startValue = 0;
	/** end value (typically in pixels) */
	private int endValue = 0;
	/** duration over which the animation takes place */
	private long durationMillis = 0;

	/**
	 * A value (difference of start and end values) that corresponds to value
	 * per millisecond
	 */
	private double valuePerMilli = 0.0;

	/** The ctm of the last performed animation operation */
	private long startMillis;

	private Timer timer;

	private double value = 0;

	/**
	 * Constructor where you specify <i>time</i> between the two pixel values.
	 */
	public Animation( int startValue, int endValue, int durationMillis ) {
		this.startValue = startValue;
		this.endValue = endValue;
		this.durationMillis = durationMillis;

		// create the value per millis.
		this.valuePerMilli = ( ( double ) ( endValue - startValue ) ) / ( ( double ) durationMillis );
	}

	/**
	 * Constructor where you specify <i>value/ms</i> between the two pixel
	 * values.
	 */
	public Animation( int startValue, int endValue, double valuePerMilli ) {

		this.startValue = startValue;
		this.endValue = endValue;
		this.valuePerMilli = valuePerMilli;
	}

	public Animation( int durationMs ) {
		this.durationMillis = durationMs;
	}

	@Override
	public void actionPerformed( ActionEvent arg0 ) {
		//get ctm
		long ctm = System.currentTimeMillis();

		//get difference of this ctm with the last ctm
		double millisPassed = ( double ) ( ctm - startMillis );

		// This may be 0 if millisPassed is small enough
		double i = millisPassed * valuePerMilli;
		if ( i == 0 ) {
			throw new RuntimeException("CRITICAL ERROR : Animation broken : incrementing by zero... so infinite loop");
		}

		value += i;

		//replace old ctm with new one.
		this.startMillis = ctm;

		if ( startValue < endValue && value >= endValue ) {
			value = Math.min( value, endValue );
			render( ( int ) value );
			stop();
		} else if ( startValue > endValue && value <= endValue ) {
			value = Math.max( value, endValue );
			render( ( int ) value );
			stop();
		} else {
			render( ( int ) value );
		}
	}

	public void start() {
		this.startMillis = System.currentTimeMillis();
		this.value = startValue;
		timer = new Timer( 50, this );
		starting();
		timer.restart();
	}

	public void stop() {
		timer.stop();
		stopped();
	}

	protected abstract void render( int value );

	/**
	 * Optional starting method for an animation. Ie. before animation commences
	 * code to initialise
	 * the animation can be placed here.
	 */
	protected void starting() {}

	/**
	 * Optional stop method for an animation. Ie. clean up code, or rendering
	 * code that happens at the end of the animation
	 */
	protected void stopped() {}

	public int getEndValue() {
		return endValue;
	}

	public void setEndValue( int endValue ) {
		this.endValue = endValue;
		if ( durationMillis > 0 ) {
			this.valuePerMilli = ( ( double ) ( endValue - startValue ) ) / ( ( double ) durationMillis );
		}
	}

	public int getStartValue() {
		return startValue;
	}

	public void setStartValue( int startValue ) {
		this.startValue = startValue;

		if ( durationMillis > 0 ) {
			this.valuePerMilli = ( ( double ) ( endValue - startValue ) ) / ( ( double ) durationMillis );
		}
	}

}
