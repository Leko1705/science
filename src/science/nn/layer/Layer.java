package science.nn.layer;

public interface Layer {

    enum Kind {
        LINEAR,
        FILTERED
    }

    Kind getKind();

    void forward();

    void backward();

    Shape getShape();


    void connectPrevious(Layer layer);


}
