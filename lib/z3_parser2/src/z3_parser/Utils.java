package z3_parser;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Utils {
	public static final int UNDEFINED_TO_INT = -9999;
	public static final float UNDEFINED_TO_DOUBLE = -9999;

	public static int toInt(String str) {
		/**
		 * Remove bracket from negative number. Ex: Convert (-2) into -2
		 */
		str = str.replaceAll("\\((" + IRegex.NUMBER_REGEX + ")\\)", "$1");

		/**
		 *
		 */
		boolean isNegative = false;
		if (str.startsWith("-")) {
			str = str.substring(1);
			isNegative = true;
		} else if (str.startsWith("+"))
			str = str.substring(1);
		/**
		 *
		 */
		int n;
		try {
			n = Integer.parseInt(str);
			if (isNegative)
				n = -n;
		} catch (Exception e) {
			n = Utils.UNDEFINED_TO_INT;
		}
		return n;
	}

	public static double toDouble(String str) {
		/**
		 * Remove bracket from negative number. Ex: Convert (-2) into -2
		 */
		str = str.replaceAll("\\((" + IRegex.NUMBER_REGEX + ")\\)", "$1");

		/**
		 *
		 */
		boolean isNegative = false;
		if (str.startsWith("-")) {
			str = str.substring(1);
			isNegative = true;
		} else if (str.startsWith("+"))
			str = str.substring(1);
		/**
		 *
		 */
		double n;
		try {
			n = Double.parseDouble(str);
			if (isNegative)
				n = -n;
		} catch (Exception e) {
			n = Utils.UNDEFINED_TO_DOUBLE;
		}
		return n;
	}

	/**
	 * @see #{CustomJevalTest.java}
	 * @param expression
	 * @return
	 */
	public static String transformFloatNegativeE(String expression) {
		Matcher m = Pattern.compile("\\d+E-\\d+").matcher(expression);
		while (m.find()) {
			String beforeE = expression.substring(0, expression.indexOf("E-"));
			String afterE = expression.substring(expression.indexOf("E-") + 2);

			String newValue = "";

			if (Utils.toInt(afterE) != Utils.UNDEFINED_TO_INT) {
				int numDemicalPoint = Utils.toInt(afterE);

				if (numDemicalPoint == 0) {
					newValue = beforeE;

				} else if (beforeE.length() > numDemicalPoint) {
					for (int i = 0; i < beforeE.length() - numDemicalPoint; i++)
						newValue += beforeE.toCharArray()[i];
					newValue += ".";

					for (int i = beforeE.length() - numDemicalPoint; i < beforeE.length(); i++) {
						newValue += beforeE.toCharArray()[i];
					}
				} else {
					newValue += "0.";
					for (int i = 0; i <= numDemicalPoint - 1 - beforeE.length(); i++) {
						newValue = newValue + "0";
					}
					newValue = newValue + beforeE;
				}
			}

			expression = expression.replace(m.group(0), newValue);
		}
		return expression;
	}

	public static String transformFloatPositiveE(String expression) {
		Matcher m = Pattern.compile("\\d+E\\+\\d+").matcher(expression);
		while (m.find()) {
			String beforeE = expression.substring(0, expression.indexOf("E+"));
			String afterE = expression.substring(expression.indexOf("E+") + 2);

			String newValue = "";

			if (Utils.toInt(afterE) != Utils.UNDEFINED_TO_INT) {
				int numDemicalPoint = Utils.toInt(afterE);

				if (numDemicalPoint == 0) {
					newValue = beforeE;

				} else {
					newValue = beforeE;
					for (int i = 0; i < numDemicalPoint; i++)
						newValue += "0";
				}
			}

			expression = expression.replace(m.group(0), newValue);
		}
		return expression;
	}

	/**
	 * Doc noi dung file
	 *
	 * @param filePath duong dan tuyet doi file
	 * @return noi dung file
	 */
	public static String readFileContent(String filePath) {
		StringBuilder fileData = new StringBuilder(3000);
		try {
			BufferedReader reader;
			reader = new BufferedReader(new FileReader(filePath));
			char[] buf = new char[10];
			int numRead = 0;
			while ((numRead = reader.read(buf)) != -1) {
				String readData = String.valueOf(buf, 0, numRead);
				fileData.append(readData);
				buf = new char[1024];
			}
			reader.close();
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			return fileData.toString();
		}
	}

	/**
	 * Tao folder
	 *
	 * @param path
	 */
	public static void createFolder(String path) {
		File destDir = new File(path);
		if (!destDir.exists())
			destDir.mkdir();
	}

	public static void writeContentToFile(String content, String filePath) {
		try {
			Utils.createFolder(new File(filePath).getParent());
			PrintWriter out = new PrintWriter(filePath);
			out.println(content);
			out.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
}
