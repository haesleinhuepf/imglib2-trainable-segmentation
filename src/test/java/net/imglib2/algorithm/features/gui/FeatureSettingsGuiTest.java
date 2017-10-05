package net.imglib2.algorithm.features.gui;

import net.imagej.ops.OpService;
import net.imglib2.algorithm.features.FeatureGroup;
import net.imglib2.algorithm.features.Features;
import net.imglib2.algorithm.features.GlobalSettings;
import net.imglib2.algorithm.features.GroupedFeatures;
import net.imglib2.algorithm.features.gson.FeaturesGson;
import org.scijava.Context;

import javax.swing.*;

/**
 * @author Matthias Arzt
 */
public class FeatureSettingsGuiTest {

	public static void main(String... args) throws InterruptedException {
		Context context = new Context(OpService.class);
		OpService ops = context.service(OpService.class);
		GlobalSettings settings = GlobalSettings.defaultSettings();
		FeatureGroup fg = Features.group(ops, settings, GroupedFeatures.gauss());
		FeatureSettingsGui gui = new FeatureSettingsGui(context, fg);
		showFrame(gui.getComponent());
		System.out.println(FeaturesGson.toJsonTree(fg));
	}

	private static void showFrame(JComponent component) throws InterruptedException {
		JFrame frame = new JFrame("Change Feature Settings Demo");
		frame.add(component);
		frame.setSize(300, 300);
		frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		frame.setVisible(true);
		while(frame.isVisible()) {
			Thread.sleep(100);
		}
	}
}
