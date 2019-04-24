package learn.nn.core;

/**
 * A PerceptronUnit is a Unit that uses a hard threshold
 * activation function.
 */
public class PerceptronUnit extends NeuronUnit {
	
	/**
	 * The activation function for a Perceptron is a hard 0/1 threshold
	 * at z=0. (AIMA Fig 18.7)
	 */
	@Override
	public double activation(double z) {
		// This must be implemented by you
		if (z >= 0) {
			return 1;
		}
		return 0;
	}

	@Override
	public double activationPrime(double z) {
		return 0;
	}

	/**
	 * Update this unit's weights using the Perceptron learning
	 * rule (AIMA Eq 18.7).
	 * Remember: If there are n input attributes in vector x,
	 * then there are n+1 weights including the bias weight w_0. 
	 */
	@Override
	public void update(double[] x, double y, double alpha) {
		double output = h_w(x);
		for (int i = 0; i < x.length + 1; i++) {
			if (i == 0) {
				setWeight(i, getWeight(i) + alpha * (y - output) * 1);
			} else {
				setWeight(i, getWeight(i) + alpha * (y - output) * x[i - 1]);
			}
		}
	}
}
