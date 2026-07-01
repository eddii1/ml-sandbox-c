# C++ ML SANDBOX

Proiectul este un sandbox mic pentru modele ML in C++. Ideea este sa avem o structura simpla de OOP, nu o aplicatie mare.

Aplicatia lucreaza cu un dataset impartit in `train` si `test`, apoi ruleaza modele diferite prin aceeasi interfata de baza `Model`.

## Cum ruleaza

Build:

```bash
cmake --build build
```

Run:

```bash
./build/MLSANDBOX
```

Demo-ul foloseste fisierele:

```text
examples/linear_regression/train
examples/linear_regression/test
```

Formatul fiecarei linii este:

```text
label,feature1,feature2,...
```

Exemplu:

```text
3,1
5,2
7,3
```

Asta inseamna ca primul numar este raspunsul asteptat, iar restul sunt feature-uri.

## Clase principale

### `Sample`

`Sample` tine datele efective:

```cpp
std::vector<float> v;
std::vector<std::vector<float>> m;
```

`v` este vectorul de labels/output-uri. `m` este matricea de feature-uri.

Clasa mai are metode care convertesc datele in Armadillo:

```cpp
arma::rowvec responses_as_rowvec() const;
arma::Row<size_t> responses_as_labels() const;
arma::mat predictors_as_mat() const;
```

Aceste conversii sunt necesare pentru ca `mlpack` lucreaza cu tipurile din Armadillo.

### `Dataset`

`Dataset` foloseste compunere:

```cpp
class Dataset {
    Sample train, test;
};
```

Deci un dataset are doua sample-uri: unul pentru antrenare si unul pentru testare.

### `DatasetLoader`

`DatasetLoader` citeste fisierele `train` si `test` dintr-un folder.

Acum foloseste exceptii proprii. Daca fisierul lipseste sau o linie este invalida, arunca `DatasetException`.

### `Model`

`Model` este clasa de baza abstracta pentru modelele ML.

```cpp
class Model {
public:
    virtual ~Model() = default;
    virtual std::shared_ptr<Model> clone() const = 0;

    void train(const Dataset& dataset);
    void predict(const Dataset& dataset);
    void evaluate(const Dataset& dataset);

protected:
    virtual void train_impl(const Sample& sample) = 0;
    virtual void predict_impl(const Sample& sample) = 0;
    virtual void evaluate_impl(const Sample& sample) = 0;
};
```

Aceasta este o forma de interfata non-virtuala: functiile publice sunt `train`, `predict`, `evaluate`, iar clasele derivate implementeaza detaliile in `train_impl`, `predict_impl`, `evaluate_impl`.

### `LinearRegression` si `KNN`

Aceste clase mostenesc `Model`.

```cpp
class LinearRegression : public Model { ... };
class KNN : public Model { ... };
```

Ele suprascriu functiile virtuale pure si pot fi folosite prin pointer la clasa de baza:

```cpp
std::shared_ptr<Model> model = ModelFactory::linear_regression();
model->train(dataset);
model->predict(dataset);
model->evaluate(dataset);
```

Aceasta este partea principala de polimorfism.

## Exceptii

Fisierul `AppExceptions.h` contine ierarhia proprie de exceptii:

```cpp
class AppException : public std::runtime_error { ... };
class DatasetException : public AppException { ... };
class ModelException : public AppException { ... };
```

Exemple de folosire:

```cpp
if (!fin.is_open())
    throw DatasetException("Nu pot deschide fisierul ...");
```

```cpp
if (!trained)
    throw ModelException("Modelul nu este antrenat.");
```

In `main`, exceptiile sunt prinse asa:

```cpp
try {
    // cod normal
} catch (const AppException& err) {
    std::cerr << err.what() << '\n';
} catch (const std::exception& err) {
    std::cerr << err.what() << '\n';
}
```

Astfel codul normal este separat de codul de tratare a erorilor.

## Smart pointers si `clone`

Proiectul foloseste `std::shared_ptr<Model>` pentru a tine obiecte derivate prin pointer la clasa de baza.

Pentru copiere corecta, `Model` are metoda virtuala `clone`:

```cpp
virtual std::shared_ptr<Model> clone() const = 0;
```

In `LinearRegression`:

```cpp
std::shared_ptr<Model> LinearRegression::clone() const {
    return std::make_shared<LinearRegression>(*this);
}
```

In `KNN`:

```cpp
std::shared_ptr<Model> KNN::clone() const {
    return std::make_shared<KNN>(*this);
}
```

Aceasta evita object slicing si permite copierea corecta a unui model fara sa stim tipul concret.

## Design pattern: Factory

`ModelFactory` este un Factory simplu.

```cpp
class ModelFactory {
public:
    static std::shared_ptr<Model> linear_regression();
    static std::shared_ptr<Model> knn(std::size_t k = 3);
};
```

Scopul lui este sa ascunda detaliile de creare:

```cpp
auto model = ModelFactory::linear_regression();
auto knn = ModelFactory::knn(3);
```

Codul care foloseste modelul nu trebuie sa stie exact daca obiectul este `LinearRegression` sau `KNN`.

## Design pattern secundar: Strategy

`Model` + clasele derivate formeaza si un Strategy pattern simplu.

`Experiment` poate primi orice model care respecta interfata `Model`:

```cpp
Experiment exp("demo", dataset, ModelFactory::linear_regression());
```

Sau:

```cpp
Experiment exp("demo", dataset, ModelFactory::knn(3));
```

Algoritmul se schimba, dar clasa `Experiment` ramane aproape la fel.

## Clasa `Experiment`

`Experiment` este o clasa compusa din:

```cpp
std::string name;
Dataset dataset;
std::shared_ptr<Model> model;
History<double> scores;
```

Deci un experiment are un nume, un dataset, un model si un istoric de scoruri.

Aceasta clasa arata cerinta cu pointer la baza ca atribut al altei clase:

```cpp
std::shared_ptr<Model> model;
```

Functia `run` foloseste apeluri virtuale prin pointer la baza:

```cpp
model->train(dataset);
model->predict(dataset);
model->evaluate(dataset);
```

### Copy and swap

`Experiment` implementeaza copiere prin `clone` si operator de atribuire cu copy-and-swap:

```cpp
Experiment::Experiment(const Experiment& other) {
    model = other.model->clone();
}

Experiment& Experiment::operator=(Experiment other) {
    swap(*this, other);
    return *this;
}
```

Acest lucru este util pentru ca `model` este un pointer la clasa de baza.

## Template

Fisierul `History.h` contine o clasa template simpla:

```cpp
template <typename T>
class History {
    std::vector<T> values;
public:
    void add(const T& value);
    std::size_t size() const;
    const std::vector<T>& all() const;
};
```

In proiect este folosita ca:

```cpp
History<double> scores;
```

Adica experimentul salveaza scorurile numerice obtinute dupa evaluare.

## Dynamic cast

`Experiment::print_model_details` foloseste `dynamic_cast` pentru a afisa detalii specifice modelului concret:

```cpp
if (const auto* lr = dynamic_cast<const LinearRegression*>(model.get())) {
    std::cout << lr->get_last_r2();
} else if (const auto* knn = dynamic_cast<const KNN*>(model.get())) {
    std::cout << knn->get_last_accuracy();
}
```

Acesta este un caz acceptabil deoarece `get_last_r2` exista doar pe `LinearRegression`, iar `get_last_accuracy` exista doar pe `KNN`.

Nu este folosit pentru logica principala. Logica principala ramane virtuala prin `Model`.

## Armadillo pe scurt

Armadillo este biblioteca de algebra liniara folosita de `mlpack`.

Tipuri importante:

```cpp
arma::mat        // matrice de double
arma::rowvec     // vector linie de double
arma::Row<size_t> // vector linie pentru labels intregi
```

In `Sample::predictors_as_mat`, matricea este construita asa:

```cpp
arma::mat result(rows, sample_count());
```

La `mlpack`, coloanele sunt sample-urile, iar randurile sunt feature-urile.

## Cerinte bifate

- Clase separate in `.h` si `.cpp`.
- Mostenire: `LinearRegression` si `KNN` mostenesc `Model`.
- Functii virtuale pure: `train_impl`, `predict_impl`, `evaluate_impl`, `clone`.
- Destructor virtual in baza `Model`.
- Pointer la clasa de baza: `std::shared_ptr<Model>` in `Experiment`.
- Constructor virtual: `clone`.
- Copy constructor si copy-and-swap in `Experiment`.
- Exceptii proprii: `AppException`, `DatasetException`, `ModelException`.
- `try`/`catch` in `main`.
- `dynamic_cast` in `Experiment::print_model_details`.
- Template: `History<T>`.
- Design pattern: `ModelFactory` ca Factory.
- Design pattern secundar: `Model` ca Strategy.

## Observatie

Codul este intentionat pastrat simplu. Scopul este sa arate conceptele cerute pentru tema/test, nu sa fie o librarie ML completa.
