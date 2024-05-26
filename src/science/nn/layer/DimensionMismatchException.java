package science.nn.layer;

public class DimensionMismatchException extends RuntimeException {

    public DimensionMismatchException(String msg){
        super(msg);
    }

    public DimensionMismatchException(){
        super();
    }

}
