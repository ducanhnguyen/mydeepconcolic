package z3_parser;

/**
 * Generate static solution of a test path
 *
 * @author ducanhnguyen
 */
public interface IStaticSolutionGeneration extends IGeneration {
    final String NO_SOLUTION = "";
    final String EVERY_SOLUTION = " ";

    /**
     * Generate static solution
     *
     * @return
     * @throws Exception
     */
    void generateStaticSolution() throws Exception;
    /**
     * Get static solution
     *
     * @return
     */
    String getStaticSolution();

}