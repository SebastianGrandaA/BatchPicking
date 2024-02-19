# BatchPicking

## Abstract
    TODO integrar con BatchPicking.py
    El problema de batch picking consiste en agrupar un conjunto de ordenes en lotes, y luego definir la secuencia de ubicaciones de almacenamiento a visitar para recuperar todas las ordenes asignadas al lote.
    El objetivo es minimizar la distancia total recorrida por los pickers, y maximizar la cercania entre las ordenes dentro de un lote.
        
    Dos approaches para resolver el problema de batch picking en un warehouse.
    El primer enfoque es el enfoque conjunto, que resuelve el problema como un vehicle routing problem (VRP).
    El segundo enfoque es una aproximacion del conjunto, que resuelve el problema como un problema de particionamiento, y posteriormente un travelling salesman problem (TSP) por cada particion.
    En esta presentacion, explicare la formalizacion del problema, las soluciones propuestas, y los resultados obtenidos.


    The algorithms were implemented in Python and the code is available in the [repository]().
        Architecture design
            Principles: (not escalability)
            Modularity: to extend

            Design patterns: inherit, etc..

    Mencionar: offline VRP (no hay restriccion de tiempo de ejecucion)


## Architecture
    def solve methods are the interface of the optimization process. It then can be further call several methods, which normally is more specific (route, batch, etc)
    ****
    Explicar arquitectura del codigo
    La logica se separa en dos componentes: domain logic and services.
    The domain logic contains the models, ....
    The services contains the details that are not 
    This separation is done to make the code more modular, testable, and maintainable.


    La estructura del proyecto esta inspirada in the modularity priciple (en el patron de diseno: (Hexagonal Architecture, Clean Architecture, Onion Architecture, , ports and adapters, etc)
    En ese sentido, el codigo fuente (src) esta organizado en tres carpetas independientes: app, domain, services.
    La carpeta app contiene la ejecucion de los casos de uso, como el de `optimize`, `experiment`, y `describe`.
        El caso de uso `optimize` resuelve el problema de optimizacion para una instancia en particular, mientras que el caso de uso `experiment` resuelve multiples instancias. Obtenido los resultados, el caso de uso `describe` analiza los resultados obtenidos con estadisticas y graficas.
    La carpeta domain contiene la logica del negocio, como los modelos matematicos, y la logica de optimizacion.
        Dos enfoques han sido implementados: el enfoque conjunto y el enfoque secuencial. El enfoque conjunto resuelve el problema de optimizacion como un vehicle routing problem (VRP), mientras que el enfoque secuencial resuelve el problema como un problema de particionamiento, y posteriormente un travelling salesman problem (TSP) por cada particion.
        En especifico, para el enfoque conjunto, se ha implementado un capacitated vehicle routing problem with pickup and delivery (CVRPPD) con la libraria [OR-Tools](https://developers.google.com/optimization/routing).
        ... detailles de implementacion, beneficios, TRUCOS !....

    La carpeta services contiene la logica de implementacion, como la lectura y escritura de archivos, la validacion de soluciones, y la ejecucion de experimentos (benchmark).
        La validacion de soluciones fue tomado del [repositorio del proyecto](https://gitlab.com/LinHirwaShema/poip-2024), sobre el cual no se ha aplicado ningun cambio significativo.



        * Justification of the language used
            Python: versatile
            Julia: recomendar en lugar de python

        * List of modules and libraries
        * Architeture of the code
            * Classes
            * Functions
            * Modules
            * Packages
            * Files
            * etc


        Python es un lenguaje versatil pero poco eficiente para aplicaciones de alto rendimiento.
        A este problema se le conoce coomo el Two-Language problem, en donde se utiliza un lenguaje facil de usar para un prototipo y luego se pasa a un lenguaje mas eficiente para la implementacion final.
        Julia es un lenguaje que se ha vuelto popular para este proposito, ya que es facil de usar como Python pero tiene un rendimiento similar a C.
        Por lo tanto, se recomienda que este proyecto sea migrado a Julia para mejorar su rendimiento.
        Otra ventaja de julia es multiple dispatch, que permite que el codigo sea mas modular, facil de extender y mas eficiente por el JIT compiler.
        En especifico, la decision de utilizar python para este proyecto fueron estrictamente por el requerimiento del proyecto, en el cual se pedia que se utilizara python o C++.







    Se han implementado dos casos de uso: el de optimizar una instancia en particular, y el de ejecutar un benchmark.
    El primer caso de uso es el mas simple, y se puede ejecutar con el siguiente comando:
        make optimize -m joint -p data/A_data_2023-05-27 -t 60

    El segundo caso de uso es mas complejo, y se puede ejecutar con el siguiente comando:
        make benchmark -m joint -p data/A_data_2023-05-27, data/A_data_2023-05-22, data/A_data_2023-05-25 -t 60

    Donde -m es el modelo a utilizar, -p es la ruta de la instancia, y -t es el tiempo maximo de ejecucion en segundos.




This project aims to solve the BatchPicking problem and to compare the **joint** and the **sequential** solutions.
The complete description of the problem and the proposed solution can be found in the [documentation](https://www.overleaf.com/read/xfgcnzwccnqj#8fe7b9). A rendered version is available at `docs/` folder.

To set up this project, first create a virtual environment and install the dependencies using the following command:

```bash
$ make init
```

## Usage

For regular usage, activate the virtual environment:
    
```bash
$ make start
```

There are three use cases: optimize, experiment, and describe.

The **optimize use case** is used to solve a single instance:

```bash
$ make optimize
```

The **experiment use case** is used to solve multiple instances:

```bash
$ make experiment
```

To run all instances, use the following command:

```bash
$ make experiment_all
```

The **describe use case** is used to analyze the results of the optimization process:

```bash
$ make describe
```

Finally, to run them all, use:

```bash
$ make test
```

## Develop

Before pushing a commit, ensure to format the code and export the dependencies:

```bash
$ make format
$ make freeze
```

## Notes

The provided data contains three files outside to their corresponding instance folders (`'adjacencyMatrix.txt', 'aisleSubdivision.txt', 'positionList.txt'`). Therefore, a pre-process is required to add those files to the instance folders. For **new instances**, edit the `src/services/scripts/duplicate_files.py` script and run:

```bash
$ make pre-process
```
