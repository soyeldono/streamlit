import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from herramientas import to_basic, card_filter
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpmax, fpgrowth
import base64
from sklearn.cluster import KMeans
from PIL import Image

def load_data(data_set="card"):
    if data_set == "card":
        data_card = pd.read_csv("./base_datos/listacartas_basedatos/CardList_DataBase.csv")
        data_card = data_card.dropna(axis=0)
        return data_card
    elif data_set == "tier":
        data_tier = pd.read_csv("./base_datos/tierlist_basedatos/Tier_DataBase.csv")
        data_tier = data_tier.dropna(axis=1)
        return data_tier
    else:
        raise TypeError("no existe " + str(data_set) + ".")



st.sidebar.title("- Temas del Proyecto -")
choise = st.sidebar.selectbox("",("Inicio","Análisis","Herramientas","YuGi-Oh Duel Links","Rullings"),key=0)

data_card = load_data()
data_tier = load_data("tier")

if choise == "Inicio":
    st.title("Proyecto Final de Mineria de Datos")
    st.markdown("Donovan Mosheh Ramirez Trejo - UNAM ENES Morelia")
    st.markdown("## Aplicando Algoritmos de Mineria de Texto y Aprendizaje no Supervisado para la predicción del Meta en el juego de Cartas YuGi-Oh Duel Links")
    

    st.subheader("Introducción")
    st.write(
        """En el juego de cartas de YuGi-Oh siempre ha estado la pregunta de ¿Que hace un deck Meta?
        ¿Qué tiene de especial este deck que los démas no? Por ello en esto proyecto se harán usos de
        técnicas como Reglas de Asociación y K-Means. Junto al meta y estilo del juego durante ese periodo
        para intentar predecir si una carta es meta o no.
        """
        )
    
    st.subheader("Recomendaciones")
    st.write(
        """
        Si eres una persona la cual nunca ha jugado al juego de cartas YuGi-Oh o no lo has jugado e un largo
        periodo de tiempo, se recomienda fuertemente ir primero a la pestaña de 'YuGi-Oh Duel Links' para 
        saber de que trata el formato digital del juego de cartas. Después se recomienda ir la pesataña de 
        'Rullings' para entender conceptos del juego que se tocan en este proyecto (ambos se encuentran en la 
        barra lateral). Finalmente paciencia, este juego puede que al principio no entiendes nada, pero una vez
        que logras captarlo todo lo demás es muy sencillo.
        """
    )

    st.subheader("Datos")
    st.write(
        """Para este proyecto se creó dos data sets desde cero con la finalidad de poder agregar nuevos parámetros
        que otros data sets no tenian, aparte de que eran muy antiguos y escasos de información. El primer data set con 
        nombre 'CardList_DataBase.csv' una lista con 750 cartas de algunos de los decks que llegaron a ser meta
        o como minimo rogue, mas sin embargo no estan todos los decks. El segundo data set es el 'TierList_DataBase.csv' 
        contiene la tier list junto con nombre de los deck que fueron meta, sus cartas, habilidad y que tier correspondian.
        El data set 'CardList_DataBase.csv' tiene fecha máxima de actualización del 31 de Diciembre del 2020  [31/12/2020].
        Y el data set 'Tier_DataBase.csv' tiene fecha máxima de actualizació del 7 de Febrero del 2018 [0/02/2018].
        """
    )
    st.subheader("CardList_DataBase")
    st.markdown(
        """
        - Tipo de Carta: Los 3 tipos de cartas que hay ([str],int)
            - Monstruo: 0
            - Magia: 1
            - Trampa: 2

        - Nombre de Carta: Nombre de cada carta, (str)

        - Texto de la Carta: Texto que tiene cada carta (str)
            - Si hay texto: descripcion de la carta
            - Si es '#': Es un monstruo NORMAL y no importa el texto de este
        
        -  Tipo de Magia: Los 5 tipos de cartas magicas que hay ([str],int)
            - Magia Normal: 0
            - Magia de Juego Rapido: 1
            - Magia Continua: 2
            - Magia de Campo: 3
            - Magia de Ritual: 4
            - Si es '#' o -1, la carta no es de magia

        - Tipo de Trampa: Los 3 tipos de cartas trampa que hay ([str],int)
            - Trampa Normal: 0
            - Trampa Continua: 1
            - Trampa de Contra Efecto: 2
            - Si es '#' o -1, la carta no es de trampa
        
        - Tipo de Monstruo: Los 24 Tipos de monstruo que hay a la fecha de creacion de este Data Set ([str],int). 
            - Si el monstruo corresponde a Ritual,Volteo,Toon,Spirit,Geminis,etc... Estos aparte de su Tipo de Monstruo llevaran una diagonal '/' para especificar si corresponden a alguno de los ya mencionados. ejemplo: Guerrero/VOLTEO, indicando que el monstruo aparte de ser guerrero tambien corresponde a los monstruos de 'VOLTEO'. Lo mismo con los del extra deck. (Leer el .txt para saber todos los tipos de monstruos)
        
        - Atributo: Atributo del Monstruo ([str],int)
            - Agua: 0
            - Fuego: 1
            - Luz: 2
            - Oscuridad: 3
            - Tierra: 4
            - Viento: 5
            - Divinidad: 6
            - Si es '#' o -1, la carta no es un monstruo
        
        - Nivel/Rango/Escala/Enlace: Nivel/Rango/Escala/Enlace de cada monstruo (int)
            - Si es -1, la carta no es un monstruo
        
        - ATK: Ataque de cada monstruo (int)
            - Si es -1, la carta no es un monstruo o su ataque es '?'
        
        - DEF: Defensa de cada monstruo (int)
            - Si es -1, la carta no es un monstruo o su defensa es '?' 
        
        - Fecha de Publicacion de la Carta: Fecha aproximada de publicacion de la carta por primera vez o la mas antigua que encontre (Date)
        
        - Rareza: Rareza de la carta (str)

        - Metodo de Obtencion: La o las formas de obtener la carta ([str],int), **no se cuenta el ticket de ensueño UR/SR**
            - Caja Grande: 0
            - Caja Mini: 1
            - Estructura: 2 (sea EX o no)
            - Selection Box Grande: 3
            - Selection Box Mini: 4
            - Gratis: 5, Gratis quiere decir conseguir las cartas por los siguientes métodos:
                - Duelo contra un personaje/npc
                - Deck inicial
                - Subir de nivel un personaje
                - Ticket
                - Evento
                - Celebraciones
                - Cambia cartas/Cambia orbes EX
        """
    )
    st.write(data_card)
    
    st.subheader("Tier_DataBase")
    st.markdown(
        """
        - Nombre del Deck: Nombre por el cual se busca o se le conoce en la comunidad al deck (str)

        - Carta: Cartas que juega el deck (str)

            Este apartado está separado los tipos de carta, primero estan los monstruos del Main Deck
            luego las magias, después las trampas y al final los monstruos del Extra Deck. No hay 
            Side Deck. (Todos separados por **;** entre cada tipo de cartad)

        - Tier: Su posición en el meta (int)

        - Habilidad: La o las habilidades que comunmente se usa en el deck (str)

        - Fecha: La fecha aproximada donde el deck tuvo su impacto (date formato dia/mes/año)

        - Cartas con Banlist: Muestra todas las cartas que tiene en la banlist en esa fecha (str)
        """
    )

    st.write(data_tier)


    st.write("*Cada Data Set tiene su porpio archivo .txt que describe las columnas*")

elif choise == "Análisis":
    data_analisis_card = to_basic.to_minus(data_card)
    data_analisis_tier = to_basic.to_minus(data_tier,columnas=list(data_tier))

    st.title("Solucionando el Problema")

    st.subheader("Recomendaciones")
    st.write(
        """
        Para poder entender MUCHO mejor todo lo que se hace en esta seccion del proyecto, se recomienda
        fuertemente ir a la pestaña de 'Herramientas','YuGi-Oh Duel Links' y leer lo que se prsenta para
        conocer conceptos, reglas, etc... que se mencionaran en repetidas ocaciones en esta sección. Y
        para finalizar, sería tambien de mucha ayuda leer TODO el reglamento del juego. (Reglamento
        en la pestaña 'YuGi-Oh Duel Links') (En caso de querer buscar una carta en especifico, se puede
        usar la barra lateral en la parte de 'Buscar por Nombre', traerá la primer carta
        que más se parezca al texto escrito. Se buscará en la forma 'smb', más información entrar a la pestaña
        'Herramientas' en la barra lateral)
        """
    )

    st.subheader("Limpieza de Datos")
    st.write(
        """
        Antes de siquiera empezar a plantear métodos, es necesario pulir nuestro Data Set. Primero
        se cambiarán todas las letras de mayúsculas a minúsculas y también los acentos serán removidos.
        Mas sin embargo, todas las palabras que lleven los signos del PSCT o escrito en Inglés, Problem 
        Solving Card Text, no serán quitados o cambiados. Para esto se implementó el archivo 'to_basic.py'
        en la carpeta de herramientas, que nos hará esto automaticamente. (Para saber más de esta función
        ir a la barra lateral y entrar en la pestaña 'Herramientas')(Para saber sobre el PSCT ir a la barra
        lateral y entrar a la pestaña 'YuGi-Oh Duel Links')
        """
    )
    
    st.subheader("Viendo la moda del texto")
    st.write(
        """
        La mayoría de la gente dice que el texto de las mejores cartas normalmente se repiten o son similares
        por lo cual, trataré de ver si es es verdad o no. Se usará la función 'contar_pal' en la carpeta de 
        'Herramientas' para contar las palabras SIN IMPORTAR EL PSCT.
        """
    )
    cartas_tier = data_tier["Cartas"]
    c = []
    for i in cartas_tier:
        p_c = i.split(";")
        for k in p_c:
            aux = k.split(",")
            for j in aux:
                if j[1] == " ":
                    if j[2:] not in c:
                        c.append(j[2:])
                elif j[3] == " ":
                    if j[4:] not in c:
                        c.append(j[4:])

    moda = data_analisis_card[data_analisis_card["Nombre de Carta"].isin(c)]
    moda = card_filter.contar_pal(moda,PSCT=False)
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    plt.xlabel("Palabra")
    plt.ylabel("Num Repeticiones")
    ax.bar(moda.index,moda[0])
    plt.plot(range(moda.shape[0]),[np.mean(moda[0]) for _ in range(moda.shape[0])],c="r")
    st.pyplot(fig)

    st.write(
        """
        Aun que no se pueda apreciar cuales palabras tienen mayor aparicion no será necesario
        dicha palabra, debido a que el promedio (linea horizontal roja) nos muestra que la gran
        mayoría de las palabras solamente tienen 1 repeticion. Pero ante las dudas, se hizo un
        zoom a todas las palabras que tengan una cantidad de repeticiones por encima del promedio.
        """
    )
    moda_2 = moda[moda[0]>np.mean(moda[0])]
    st.write(moda_2.T)
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    plt.xlabel("Palabra")
    plt.ylabel("Num Repeticiones")
    ax.bar(moda_2.index,moda_2[0])
    st.pyplot(fig)
    
    st.write(
        """
        Y como previamente se había menciando, palabras que más se repitan en los decks meta no
        implica que toda carta que tambien la tenga sea pueda catalogar como "Carta del Meta".
        Pero eso fue INGORANDO EL PSCT, ¿Qué pasaría si se considera el PSCT? (Para entender la
        tabla de abajo ir a la funcion 'contar_pal' que se encuentra en la barra lateral pestaña 
        'Herramientas')
        """
    )
    moda = data_analisis_card[data_analisis_card["Nombre de Carta"].isin(c)]
    moda = card_filter.contar_pal(moda,PSCT=True)
    st.write(moda.T)

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    plt.xlabel("Palabra")
    plt.ylabel("Num Repeticiones")
    ax.bar(moda.index,moda["#"])
    plt.plot(range(moda.shape[0]),[np.mean(moda["#"]) for _ in range(moda.shape[0])],c="r")
    st.pyplot(fig)

    st.write(
        """
        Nuevamente, se puede apreciar que aun respetando el PSCT no cambian mucho las cosas, pero
        igualmente se aplicará zoom para poder apreciar cuales palabras estan por encima del promedio.
        """
    )
    moda_2 = moda[moda["#"]>np.mean(moda["#"])]
    st.write(moda_2.T)
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    plt.xlabel("Palabra")
    plt.ylabel("Num Repeticiones")
    ax.bar(moda_2.index,moda_2["#"])
    st.pyplot(fig)

    nombre_carta = st.sidebar.text_input("Buscar por Nombre","~")

    if nombre_carta != "~":
        c_choice = card_filter.buscar_nombre(nombre_carta,data_card,tipo_busqueda="smb")
        if len(c_choice) > 0:
            st.sidebar.write(c_choice["Nombre de Carta"].iloc[0] + ": " + c_choice["Texto de la Carta"].iloc[0])
        else:
            st.sidebar.write("No se encontraron resultados")


    st.subheader("Analizando el Texto de Carta")
    st.write(
        """
        Como se pudo observar, claramente el simple hecho de contar las palabras que aparecen no
        tiene sentido, se hará un análisis visual al leer directamente las cartas y usando el 
        modulo de PSCT tratar de entender el meta juego. (Para poder ver cada análisis usar la
        barra lateral 'Decks Meta' y seleccionar cada uno para ver su procedimiento de análisis)
        """
    )
    decks = []
    for i in data_analisis_tier["Nombre del Deck"]:
        if i not in decks and i != "dragon zombi de ojos rojos":
            decks.append(i)
    decks.sort()
    choice_arquetipo = st.sidebar.selectbox("Decks Meta",decks)
    if choice_arquetipo == "dragon negro de ojos rojos con zombie":
        choice_arquetipo = "ojos rojos zombi"
    
    data_choice = card_filter.buscar_nombre(choice_arquetipo,data_analisis_card,tipo_busqueda="sme")
    st.write("Deck seleccionado: **" + str(choice_arquetipo) + "**, nombres de cartas similires a **" + str(choice_arquetipo) + "**:")
    st.write(data_choice)

    cartas = ""
    t1 = []
    for i,j,f in zip(data_choice["Nombre de Carta"],data_choice["Texto de la Carta"],data_choice["Fecha de Publicación de la Carta"]):
        cartas += "**Texto de la carta, " + str(i) + "**:\n " + (str(j) if j != "#" else "(Monstruo Normal)")
        cartas += " Fecha aproximada de la publicación de esta carta: [" + str(f) + "]"
        cartas += "\n\n"
        t1.append(i)
    st.write(cartas)

    st.write("**Otras cartas que tambien se juegan**")

    cartas = ""
    if choice_arquetipo == "ojos rojos zombi":
        choice_arquetipo = "dragon negro de ojos rojos con zombie"
    deck_t = data_analisis_tier[data_analisis_tier["Nombre del Deck"] == choice_arquetipo]

    c = []
    for i in deck_t["Cartas"]:
        p_c = i.split(";")
        for k in p_c:
            aux = k.split(",")
            for j in aux:
                if j[1] == " ":
                    if j[2:] not in c and j[2:] not in t1:
                        c.append(j[2:])
                elif j[3] == " ":
                    if j[4:] not in c and j[4:] not in t1:
                        c.append(j[4:])

    cartas_t = data_analisis_card[data_analisis_card["Nombre de Carta"].isin(c)]

    for i,j,f in zip(cartas_t["Nombre de Carta"],cartas_t["Texto de la Carta"],cartas_t["Fecha de Publicación de la Carta"]):
        cartas += "**Texto de la carta, " + str(i) + "**:\n " + (str(j) if j != "#" else "(Monstruo Normal)")
        cartas += " Fecha aproximada de la publicación de esta carta: [" + str(f) + "]"
        cartas += "\n\n"
    st.write(cartas)

    cont_p = data_analisis_card[data_analisis_card["Nombre de Carta"].isin(c+t1)]
    cont_p = card_filter.contar_pal(cont_p,PSCT=True)

    st.write(cont_p)

    if choice_arquetipo == "bestia gladiador":

        modas = {}
        for i in cont_p["PSCT"]:
            if i in modas:
                modas[i] += 1
            elif i not in modas:
                modas[i] = 1
        
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        plt.xlabel("PSCT")
        plt.ylabel("Num Repeticiones")
        plt.title("Cantidad de Palabras que se repiten basados en el PSCT")
        ax.bar(modas.keys(),modas.values())
        st.pyplot(fig)

        st.write(
            """
            La idea de este deck es muy sencilla, aguantar los ataques del rival con cartas
            como 'medio cerrado' para que los monstruos no sean destruidos y poder activar
            sus efectos, ya que como se puede observar tanto en los efectos de las propias
            cartas como en el modulo de PSCT, la mayoria de sus cartas dependen de su
            condicion de activación y su coste, son muy dependientes de estos dos últimos.
            Por lo tando ya tenemos una idea de como funciona el deck, pero toca ver que 
            se puede hacer con las reglas de asociacion. Se utilizará el método 'lift',
            se escogió dicho método por que 'lift' ayuda a medir si el conjunto de palabras
            es superior a lo normal entonces quiere decir que hay una relacion entre ellos.
            """
        )

        te = TransactionEncoder()
        txt = []
        excluir = ["de","que","en","y","tu","la","el","o","al","los","a"]
        caracteres = {"!":"","¡":"","\"":"","#":"","(":"",")":"","'":"",";":"",":":"","-":"",".":""}
        for i in cartas_t["Texto de la Carta"]:
            aux = []
            for k in i.split(): 
                _pal = ""
                for s in k:
                    if s in caracteres.keys():
                        _pal += caracteres[s]
                    else:
                        _pal += s
                if _pal not in txt and _pal not in excluir:
                    aux.append(_pal)
            txt.append(aux)
        te_ary = te.fit(txt).transform(txt)
        df = pd.DataFrame(te_ary,columns=te.columns_)
        frequent_itemsets = fpgrowth(df, min_support=0.35, use_colnames=True)
        st.write(frequent_itemsets)
        assc = association_rules(frequent_itemsets, metric="lift", min_threshold=1.5)
        st.write(assc)
        #st.write("ñslakdjfñalskjdf")
        #st.write(list(assc["antecedents"]))

        st.write(
            """
            Y las reglas de asociacón nos vuelven a presentar que la principal parte de este deck
            es el coste de sus cartas. Ya que como se sabe, dichas cartas dependen de primero atacar
            o ser atacadas, haber sobervivido a dicho ataque y hasta entonces se podran activar sus 
            efectos, la capacidad de selecionar cartas ya sea para destruir cartas magicas/trampas o 
            monstruos. Lo cual es bastante bueno, mas sin embargo este deck nunca fue tier 1,
            ¿Por qué no lo fue? si el deck tiene la capacidad de quitar cartas molestas ya sea bajando
            el atk o destruyendolas, aparte de que podia invocar monstruos de fusion sin polimerización.
            Bueno la respuesta nos la esta dando las propias reglas de asociación, dependen de pasar por
            2 etapas antes de activar sus grandes efectos. La condición de activación y el coste, ambos
            tienen que resolverse primero y eso hace lento, recuerda que tiene que atacar o haber recibido
            un ataque, por eso las cartas trampas que impiden que tu monstruo sea destruido. El problema
            es que decks como 'Ciber Angel' NO DESTRUYEN AL MONSTRUO, LO ENVIAN AL CEMENTERIO inhabilitando
            por completo sus efectos y dejandote vendido por el resto del juego.

            \n\n
            **Ejemplo del funcionamiento del deck**
            """
        )
        file_open = open("./imagenes_videos/gb.gif","rb")
        cont = file_open.read()
        url = base64.b64encode(cont).decode("utf-8")
        file_open.close()

        st.markdown(f'<img src="data:image/gif;base64,{url}" alt="beastia_gladiator">',
                    unsafe_allow_html=True)
        
        st.write("*Video de Jaden Sword*")

    elif choice_arquetipo == "ciber angel":

        modas = {}
        for i in cont_p["PSCT"]:
            if i in modas:
                modas[i] += 1
            elif i not in modas:
                modas[i] = 1
        
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        plt.xlabel("PSCT")
        plt.ylabel("Num Repeticiones")
        plt.title("Cantidad de Palabras que se repiten basados en el PSCT")
        ax.bar(modas.keys(),modas.values())
        st.pyplot(fig)

        te = TransactionEncoder()
        txt = []
        excluir = ["de","que","en","y","tu","la","el","o","al","los","a"]
        caracteres = {"!":"","¡":"","\"":"","#":"","(":"",")":"","'":"",";":"",":":"","-":"",".":""}
        for i in cartas_t["Texto de la Carta"]:
            aux = []
            for k in i.split(): 
                _pal = ""
                for s in k:
                    if s in caracteres.keys():
                        _pal += caracteres[s]
                    else:
                        _pal += s
                if _pal not in txt and _pal not in excluir:
                    aux.append(_pal)
            txt.append(aux)
        te_ary = te.fit(txt).transform(txt)
        df = pd.DataFrame(te_ary,columns=te.columns_)
        frequent_itemsets = fpgrowth(df, min_support=0.5, use_colnames=True)
        st.write(frequent_itemsets)
        assc = association_rules(frequent_itemsets, metric="leverage", min_threshold=0.2)
        st.write(assc)
        

        st.write(
            """
            Oh si, llegamos al Tier 0 las poderosas 'Ciber Angel'. Nada que la grafica de barras
            no haya mostrado ya. Como se puede observar, este deck tiene una alta actividad en
            las condiciones de activacion, pero estas condiciones de activacion las podriamos
            considerar efectos, ¿Por qué? si se vuelve a revisar las reglas de asociación se
            puede observar como dichas condiciones dependen de que los monstruos sean invocados 
            por Ritual y si otra vez volvemos a ver el arquetipo de las Ciber Angeles, son un deck
            de Ritual osea que si o si sus condiciones de activacion se cumplirán, 'cuando esta carta
            es invocada por riual' ese simple texto hacen que casi siempre se activen sus efectos. Pero, 
            ¿Es por ese motivo que fueron tier 0? la respuesta es no, el hecho de que este deck siempre
            active sus efectos MUY rápido no fue el motivo de su fortaleza, si no que fueron multplies 
            cosas. Primero, la EXAGERADA capacidad de busqueda que tiene el deck, se tiene al 'pájaro
            sonico' que busca las cartas magicas de ritual, se tiene al 'senju de mil manos' que busca 
            los monstruos de ritual y las propias cartas de ritual del arquetipo junto a sus monstruos 
            de ritual en un deck de 20 cartas, ¡20! Si nos ponemos a hacer las cuentas más de la mitad
            del deck son o los monstruos de ritual con sus magias u otras cartas que las buscan. Practicamente
            la probabilidad de tener una muy buena mano en tu primer turno era de más del 60%. Pero
            espera que esto es solo el comienzo. La otra parte que ayudó mas no fue clave para que el
            deck llegara a tal punto fue que, ninguna carta pierde timing, ni una sola, haciendo que tu
            oponente esté obligado a negar tus efectos con cartas que lo permitan. Luego la capacidad de 
            recuperar material es muy fácil, 2 de las 3 cartas principales del deck permiten recuperar 
            cartas. Y para rematar dakini, en aquellas épocas era un dolor inmenso, ya que dakini afecta
            al jugador y no a las carta como las bestias gladiador. Prácticamente este deck salió muy 
            fuerte para la época en la que estaba el inicio del juego.

            \n\n
            **Ejemplo del funcionamiento del deck (post ban list)**
            """
        )

        file_open = open("./imagenes_videos/ca.gif","rb")
        cont = file_open.read()
        url = base64.b64encode(cont).decode("utf-8")
        file_open.close()

        st.markdown(f'<img src="data:image/gif;base64,{url}" alt="yiber Angels">',
                    unsafe_allow_html=True)
        
        st.write("*Video de Nash R*")

    elif choice_arquetipo == "dragon negro de ojos rojos con zombie":
        modas = {}
        for i in cont_p["PSCT"]:
            if i in modas:
                modas[i] += 1
            elif i not in modas:
                modas[i] = 1
        
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        plt.xlabel("PSCT")
        plt.ylabel("Num Repeticiones")
        plt.title("Cantidad de Palabras que se repiten basados en el PSCT")
        ax.bar(modas.keys(),modas.values())
        st.pyplot(fig)

        te = TransactionEncoder()
        txt = []
        excluir = ["de","que","en","y","tu","la","el","o","al","los","a"]
        caracteres = {"!":"","¡":"","\"":"","#":"","(":"",")":"","'":"",";":"",":":"","-":"",".":""}
        for i in cartas_t["Texto de la Carta"]:
            aux = []
            for k in i.split(): 
                _pal = ""
                for s in k:
                    if s in caracteres.keys():
                        _pal += caracteres[s]
                    else:
                        _pal += s
                if _pal not in txt and _pal not in excluir:
                    aux.append(_pal)
            txt.append(aux)
        te_ary = te.fit(txt).transform(txt)
        df = pd.DataFrame(te_ary,columns=te.columns_)
        frequent_itemsets = fpgrowth(df, min_support=0.35, use_colnames=True)
        st.write(frequent_itemsets)
        st.write(association_rules(frequent_itemsets, metric="lift", min_threshold=2.5))

        st.write(
            """
            El Red Eyes Zombie Dragon toda su estrategia trata de mandar al cementerio lo más
            rápido posible al  'Dragon Zombi de Ojos Rojos' con cartas como 'Gozuki' o la
            'Perspicacia de Ojos Rojos' esto con la finalidad de revivirlo con la carta de trampa
            'Espiritu de los ojos Rojos' también aprovechándose de cartas como 'Controlador de 
            Enemigos' para sacrificar el propio 'Dragon Zombi de Ojos Rojos' para tomar el control
            de un monstruo de nuestro rival y nuevamente revivirlo para así tener 2 monstruos en el
            campo por 2 cartas. Este deck era bastante bueno y facilmente su estrategia le permitió
            durar mucho más tiempo en el meta que 'Cyber Angels' o 'Ninja', pero el mayor defecto
            de este deck es la poca capacidad de respuesta a monstruos mas fuertes que 
            'Dragon Zombi de Ojos Rojos' o cartas que le hagan perder atk.

            \n\n
            *Ejemplo del funcionamiento del Deck*
            """
        )

        file_open = open("./imagenes_videos/rez.gif","rb")
        cont = file_open.read()
        url = base64.b64encode(cont).decode("utf-8")
        file_open.close()

        st.markdown(f'<img src="data:image/gif;base64,{url}" alt="Red Eyes Zombi">',
                    unsafe_allow_html=True)
        
        st.write("*Video de Nash R*")

    elif choice_arquetipo == "llama quimera":

        modas = {}
        for i in cont_p["PSCT"]:
            if i in modas:
                modas[i] += 1
            elif i not in modas:
                modas[i] = 1
        
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        plt.xlabel("PSCT")
        plt.ylabel("Num Repeticiones")
        plt.title("Cantidad de Palabras que se repiten basados en el PSCT")
        ax.bar(modas.keys(),modas.values())
        st.pyplot(fig)

        te = TransactionEncoder()
        txt = []
        excluir = ["de","que","en","y","tu","la","el","o","al","los","a"]
        caracteres = {"!":"","¡":"","\"":"","#":"","(":"",")":"","'":"",";":"",":":"","-":"",".":""}
        for i in cartas_t["Texto de la Carta"]:
            aux = []
            for k in i.split(): 
                _pal = ""
                for s in k:
                    if s in caracteres.keys():
                        _pal += caracteres[s]
                    else:
                        _pal += s
                if _pal not in txt and _pal not in excluir:
                    aux.append(_pal)
            txt.append(aux)
        te_ary = te.fit(txt).transform(txt)
        df = pd.DataFrame(te_ary,columns=te.columns_)
        frequent_itemsets = fpgrowth(df, min_support=0.3, use_colnames=True)
        st.write(frequent_itemsets)
        st.write(association_rules(frequent_itemsets, metric="lift", min_threshold=2.5))

        st.write(
            """
            El deck de las llamas quimericas tiene como objetivo el mandar o sacrificar sus propios
            monstruos ya sea desde la mano o deck para ganat ATK o invocar esos monstruos sacrificados
            nuevamente al campo. Una desventaja es que sus monstruos principales son de nivel 6 o mayor
            y no pueden ser invocados de modo especial por otras cartas que no sean las propias 
            'llamas quimera', por eso utilizan 'intercambio de almas' que permite usar los monstruos de
            nuestro oponente como sacrificio, también el uso de cartas como 'super apuro de cabeza' ayuda
            a que monstruos mas débiles aguanten ese turno para luego ser sacrificados y por supuesto el
            deck aprovecha demasiado la habilidad 'estrategia de arobru'.

            \n\n
            *Ejemplo del funcionamiento del Deck*
            """
        )

        file_open = open("./imagenes_videos/lq.gif","rb")
        cont = file_open.read()
        url = base64.b64encode(cont).decode("utf-8")
        file_open.close()

        st.markdown(f'<img src="data:image/gif;base64,{url}" alt="Llam Quimera">',
                    unsafe_allow_html=True)
        
        st.write("*Video de Rafa Armas*")

        
    elif choice_arquetipo == "mecanismo antiguo":

        modas = {}
        for i in cont_p["PSCT"]:
            if i in modas:
                modas[i] += 1
            elif i not in modas:
                modas[i] = 1
        
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        plt.xlabel("PSCT")
        plt.ylabel("Num Repeticiones")
        plt.title("Cantidad de Palabras que se repiten basados en el PSCT")
        ax.bar(modas.keys(),modas.values())
        st.pyplot(fig)

        te = TransactionEncoder()
        txt = []
        excluir = ["de","que","en","y","tu","la","el","o","al","los","a"]
        caracteres = {"!":"","¡":"","\"":"","#":"","(":"",")":"","'":"",";":"",":":"","-":"",".":""}
        for i in cartas_t["Texto de la Carta"]:
            aux = []
            for k in i.split(): 
                _pal = ""
                for s in k:
                    if s in caracteres.keys():
                        _pal += caracteres[s]
                    else:
                        _pal += s
                if _pal not in txt and _pal not in excluir:
                    aux.append(_pal)
            txt.append(aux)
        te_ary = te.fit(txt).transform(txt)
        df = pd.DataFrame(te_ary,columns=te.columns_)
        frequent_itemsets = fpgrowth(df, min_support=0.4, use_colnames=True)
        st.write(frequent_itemsets)
        st.write(association_rules(frequent_itemsets, metric="lift", min_threshold=1.6))

        st.write(
            """
            Ancient Gear tiene la peculiaridad de que, durante la fase de batalla el oponente no
            puede activar cartas magicas/ de trampa, haciendo que este deck por si solo ya tenga 
            una gran ventaja sobre los démas, sabiendo que en esas épocas era muy común tener cartas
            como 'muro de espejo' o 'pared de disrupción' hacían que este fuera una buena para evitar
            dichas cartas trampa. Pero otra vez, ¿Por qué no fué tier 1 o 2 si se jugaba muchas cartas de
            trampa?, la respuesta tambien nos la da las porpias reglas de asociación, si quitamos el
            efecto de que el oponente no puede activar cartas de magia/trampa durante la fase de batalla
            el deck solo tiene monstruos con alto ataque pero nada de resistencia a cartas que se pueden
            activar antes de la fase de batalla, como 'agujero trampa esclusa' o 'controlador de enemigos',
            y en segundo lugar no tiene forma de quitarse monstruos con atk mayor a 3000, si tomamos como
            base a las 'Ciber Angels', dakini supera los 3000atk algo que los mecanismos antiguos no pueden
            hacer nada en respuesta a ese atk y aún si lo tuvieran dakini mandaría al cementerio el 
            'golem de mecanismo antiguo', prácticamente fueron opacados por las 'Ciber Angels'. Después
            aproximadamente uno o año y medio después estos ya tendrían lo que les hace falta, quitarse
            esos monstruos con mayor atk al seleccionarlos y poder destruirlos, pero a la fecha del inicio
            del juego, les faltaba la capacidad de destruir para poder llegar al tier 2 como mínimo.

            \n\n
            **Ejemplo del funcionamiento del Deck**
            """
        )

        file_open = open("./imagenes_videos/ma.gif","rb")
        cont = file_open.read()
        url = base64.b64encode(cont).decode("utf-8")
        file_open.close()

        st.markdown(f'<img src="data:image/gif;base64,{url}" alt="Mecanismo Antiguo">',
                    unsafe_allow_html=True)
        
        st.write("*Video de Yuki G4ming*")
    
    else:
        modas = {}
        for i in cont_p["PSCT"]:
            if i in modas:
                modas[i] += 1
            elif i not in modas:
                modas[i] = 1
        
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        plt.xlabel("PSCT")
        plt.ylabel("Num Repeticiones")
        plt.title("Cantidad de Palabras que se repiten basados en el PSCT")
        ax.bar(modas.keys(),modas.values())
        st.pyplot(fig)

        te = TransactionEncoder()
        txt = []
        excluir = ["de","que","en","y","tu","la","el","o","al","los","a"]
        caracteres = {"!":"","¡":"","\"":"","#":"","(":"",")":"","'":"",";":"",":":"","-":"",".":""}
        for i in cartas_t["Texto de la Carta"]:
            aux = []
            for k in i.split(): 
                _pal = ""
                for s in k:
                    if s in caracteres.keys():
                        _pal += caracteres[s]
                    else:
                        _pal += s
                if _pal not in txt and _pal not in excluir:
                    aux.append(_pal)
            txt.append(aux)
        te_ary = te.fit(txt).transform(txt)
        df = pd.DataFrame(te_ary,columns=te.columns_)
        frequent_itemsets = fpgrowth(df, min_support=0.4, use_colnames=True)
        st.write(frequent_itemsets)
        st.write(association_rules(frequent_itemsets, metric="lift", min_threshold=2))

        st.write(
            """
            Renunciado fue el primer deck meta del juego, mas sin embargo nunca pudo sobre pasar 
            rogue. Es super simple la idea del deck, robarle los monstruos a tu oponente, fin. 
            El usar a kuribohs era la estrategia del principio, los kuribohs te poretegian del ataque
            y como son de nivel 1 servían perfectamente para invocarlo por ritual, y ¿Por qué nunca
            sobre pasó rogue? nuevamente, las reglas de asociación las mencionan. Solo es un monstruo
            de ritual que roba los monstruos de tu rival, no tiene defensa contra ninguna trampa o 
            magia, tampoco impide que sea seleccionado o algo por el estilo. Era muy vulnerable a todo
            tipo de cartas de defensa, reducción de atk, cambio de posición de batalla, etc... En algún
            momento se intentó usar el 'golem de lava', sacrificando 2 monstruos de rival y despues 
            rovar el golem de lava para tener de 3000 atk pero nuevamente, no tenia protecciones de nada,
            una trampa y se acabó, le reducen el atk y se queda atascado en el campo, etc... Aun así no
            quita que fue un buen deck para empezar el juego.

            \n\n
            *Ejemplo del funcionamiento del Deck*
            """
        )
        
        file_open = open("./imagenes_videos/rn.gif","rb")
        cont = file_open.read()
        url = base64.b64encode(cont).decode("utf-8")
        file_open.close()

        st.markdown(f'<img src="data:image/gif;base64,{url}" alt="Renunciado">',
                    unsafe_allow_html=True)
        
        st.write("*Video de DeulLinks Meta*")
    
    st.subheader("K-Means")

    st.write(
        """
        Ahora se va a agrupar las cartas con K-Means. Para hacer esto se le va a asignar un vector
        a cada carta donde el vector será el conteo de todas las palabras que tenga la carta.
        Se usará la forma de contar sin PSCT. Gracias al problema se sabe que forzosamente debe haber
        4 clusters, tier 1, tier 2, tier 3, rogue.
        """
    )


    cartas_tier = data_tier["Cartas"]
    c = []
    for i in cartas_tier:
        p_c = i.split(";")
        for k in p_c:
            aux = k.split(",")
            for j in aux:
                if j[1] == " ":
                    if j[2:] not in c:
                        c.append(j[2:])
                elif j[3] == " ":
                    if j[4:] not in c:
                        c.append(j[4:])

    a = None
    conteo_de_pal = []
    max_shape = [0,0]
    for k,i in enumerate(c):
        a = data_analisis_card[data_analisis_card["Nombre de Carta"].isin([i])]
        a = card_filter.contar_pal(a,PSCT=False)
        try:
            
            conteo_de_pal.append(np.array(a[0]))

            if k == 0:
                max_shape = [len(a[0]),k]

            if len(a[0]) > max_shape[0]:
                max_shape = [len(a[0]),k]


            for z,j in enumerate(conteo_de_pal):
                if len(j) < max_shape[0]:
                    dim_j = len(j)
                    dim_c = max_shape[0]
                    aux = np.resize(j,conteo_de_pal[max_shape[1]].shape)
                    aux[-abs(dim_j-dim_c):] = 0
                    conteo_de_pal[z] = aux
        except:
            
            conteo_de_pal.append(np.array(a))

            if k == 0:
                max_shape = [len(a),k]

            if len(a) > max_shape[0]:
                max_shape = [len(a),k]


            for z,j in enumerate(conteo_de_pal):
                if len(j) < max_shape[0]:
                    dim_j = len(j)
                    dim_c = max_shape[0]
                    aux = np.resize(j,conteo_de_pal[max_shape[1]].shape)
                    aux[-abs(dim_j-dim_c):] = 0
                    conteo_de_pal[z] = aux

 
    conteo_de_pal = np.array(conteo_de_pal).T
    aux = [str(i) for i in range(conteo_de_pal.shape[1])]
    aux = pd.DataFrame(conteo_de_pal,columns=aux)
    indx = []
    for i in aux:
        if np.mean(np.array(aux[str(i)])) == 0:
            indx.append(i)
    aux.drop(columns=indx,inplace=True)
    conteo_de_pal = aux.to_numpy().T
    st.write(conteo_de_pal)

    kmeans = KMeans(n_clusters=4, random_state=0).fit(conteo_de_pal)
    #st.write(kmeans.labels_)
    st.write("Identificando cada cluster")

    st.write("ciber angel benten - Tier 1")
    tier1 = card_filter.buscar_nombre("ciber angel benten",data_analisis_card,tipo_busqueda="ex")
    #print(tier1)
    a = card_filter.contar_pal(tier1,PSCT=False,axis=0)
    a = a.to_numpy()
    a = list(a[0])
    #print(a)
    for i in range(abs(len(a)-max_shape[0])):
        a.append(0)
    a = np.array([a])
    #print(a)
    st.write(kmeans.predict(a))

    st.write("gozuki - Tier 2")
    tier2 = card_filter.buscar_nombre("gozuki",data_analisis_card,tipo_busqueda="ex")
    #print(tier1)
    a = card_filter.contar_pal(tier2,PSCT=False,axis=0)
    a = a.to_numpy()
    a = list(a[0])
    #print(a)
    for i in range(abs(len(a)-max_shape[0])):
        a.append(0)
    a = np.array([a])
    st.write(kmeans.predict(a))

    st.write("golem de mecanismo antiguo - Tier 3")
    tier2 = card_filter.buscar_nombre("golem de mecanismo antiguo",data_analisis_card,tipo_busqueda="ex")
    a = card_filter.contar_pal(tier2,PSCT=False,axis=0)
    a = a.to_numpy()
    a = list(a[0])
    for i in range(abs(len(a)-max_shape[0])):
        a.append(0)
    a = np.array([a])
    st.write(kmeans.predict(a))

    st.markdown("## Predicciones con cartas del año 2018 (finales del año) - 2019")

    st.write("piedra blanca de la leyenda")
    test = card_filter.buscar_nombre("piedra blanca de la yeyenda",data_analisis_card,tipo_busqueda="ex")
    a = card_filter.contar_pal(test,PSCT=False,axis=0)
    a = a.to_numpy()
    a = list(a)
    for i in range(abs(len(a)-max_shape[0])):
        a.append(0)
    a = np.array([a])
    st.write(kmeans.predict(a))
    st.write("Real tier al momento de la salida de la carta: 3")

    st.write("seis samurais legendarios - kizan")
    test = card_filter.buscar_nombre("seis samurais legendarios - kizan",data_analisis_card,tipo_busqueda="ex")
    a = card_filter.contar_pal(test,PSCT=False,axis=0)
    a = a.to_numpy()
    a = list(a[0])
    for i in range(abs(len(a)-max_shape[0])):
        a.append(0)
    a = np.array([a])
    st.write(kmeans.predict(a))
    st.write("Real tier al momento de la salida de la carta: 1")


    st.write("señoroscuro superbia")
    test = card_filter.buscar_nombre("señoroscuro superbia",data_analisis_card,tipo_busqueda="ex")
    a = card_filter.contar_pal(test,PSCT=False,axis=0)
    a = a.to_numpy()
    a = list(a[0])
    for i in range(abs(len(a)-max_shape[0])):
        a.append(0)
    a = np.array([a])
    st.write(kmeans.predict(a))
    st.write("Real tier al momento de la salida de la carta: 1")

    st.subheader("Conclusiones")

    st.write(
        """
        Ya después de mucho probar se puede dar un veredicto al problema planteado al comienzo de
        este proyecto. ¿Se puede predecir si una carta será meta? La respuesta es NO, poder predecir
        usando solamente el texto de cartas anteriores no es suficiente para decidir si una carta
        hará meta a un arquetipo. Esto es debido principalmente a que por si solas, existen ciertas
        cartas las cuales no generan un gran impcato en el juego competitivo, pero en conjunto de 
        una, dos, tres o más cartas si hacen un verdadero cambio, y como se pudo observar en este
        proyecto el analizar solamente una carta no es muy, si no el verdadero reto proviene al 
        analizar a TODO el arquetipo completo. Por lo que un posible futuro proyecto sería analizar
        multiples cartas a la vez y ahora sí poder hacertar con mejor precisión el meta del juego. 
        Pero... si se analiza detalladamente todo el proyecto, se podrán fijar en algo, nunca se 
        mencionaron a las habilidades del juego, por lo que, ¿Puede una habilidad cambiar el meta?,
        estas 2 últimas observaciones se pueden tratar en futuros proyectos.
        """
    )

    st.subheader("Agradecimientos")

    st.write(
        """
        Este espacio se lo quiero dedicar a mis amigos del Discord que son MUY competitivos en el 
        juego y que me regalaron una parte de su tiempo al revisar y corroborar que los datos y la 
        información mostrada aquí sea correcta y coherente. A su vez, también me dieron su opinión 
        del proyecto y también me ayudaron a asegurar que es entendible para el lector. (Siempre que
        se haya hecho caso a las recomendaciones en el caso de que sea gente que no juegue al juego)
        """
    )



    
    


elif choise == "YuGi-Oh Duel Links":
    st.title("YuGi-Oh Duel Links")

    st.subheader("¿Qué es YuGi-Oh Duel Links?")
    st.write(
        """
        YuGi-Oh Duel Links es una adaptación del juego de cartas físico con el mismo nombre, pero a un formato
        más rápido que el original. A este se le denominó 'Speed Duels'. El juego se monta en un tablero de 
        tamaño 2 vertical 5 horizontal por jugador (a la fecha de creación de este proyecto). La primer columna del tablero 
        (de izquierda a derecha) esta reservado para el Extra Deck y las Cartas de Campo (más adelante se hablará 
        más a profundidad de esto), las columnas 2,3 y 4 son reservadas para las Cartas de Magia/Trampa y las 
        zonas donde podremos invocar nuestros monstruos. Finalmente la útima columna es donde estará el Cementerio
        y el Main Deck.
        """
    )

    basewidth = 300
    img = Image.open('./imagenes_videos/campo.jpg')
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)

    st.image(img,caption="Tablero de juego")

    st.subheader("Cartas de Monstruos")
    st.markdown(
        """
        Las Cartas de Monstruos son aquellas las cuales pueden ser puestas en las Columnas 2 a 4 en la primera
        fila (de arriba a abajo), estas cartas NO PUEDEN SER COLOCADAS en las Columnas 2 a 4 en la segunda fila
        a menos que la propia carta lo diga. Todas las Cartas de Monstruos tienen un nombre, atributo,
        nivel/rango (a la fecha de creación de este proyecto), Tipo, descripción, ataque/ATK y defensa/DEF (a la
        fecha de creación de este proyecto). Estas cartas van en el Main Deck. A la actual fecha de 24 de enero del 2021 solo existen 6 tipos de monstruos.""")
        
    st.markdown("""
            - Monstruos Normales: Estos no tienen efectos/habilidades que permitan hacer jugadas, pero normalmente
            este tipo de monstruos tienen más ATK y DEF que los Monstruos de Efecto.
        
        Ejemplo de Monstruo Normal:""")


    #basewidth = 300
    img_nm = Image.open('./imagenes_videos/Carta_Monstruo_Normal.png')
    #wpercent = (basewidth/float(img_nm.size[0]))
    #hsize = int((float(img_nm.size[1])*float(wpercent)))
    #img_nm = img_nm.resize((basewidth,hsize), Image.ANTIALIAS)

    st.image(img_nm,caption="Monstruo Normal")

    st.markdown("""
        - Monstruos de Efecto: Estos monstruos tienen habilidades que permiten al jugador hacer más jugadas. Los
        efectos de estos monstruos se dividen en 4 categorías:

            - Efecto Continuo: Siempre estan activos a menos que otra carta lo impida. Ejemplo: Monstruos con 2000
            de ATK o menos no pueden declarar un ataque.

            - Efecto de Activación: Estos efectos se pueden usar siempre y cuando se declare su activación. Ejemplo:
            Puedes Sacrificar esta carta, y después seleccionar 1 monstruo en el Campo; destruye ese objetivo.

            - Efecto Rápido: Estos efectos se pueden activar aún durante el turno de tu adversario. Estos tipos de
            efectos tienen una Velocidad de Hechizo 2 (revisar el manual del juego para más detalles de Velocidad de
            Hechizo, https://img.yugioh-card.com/es/rulebook/es.pdf). Ejemplo: Durante el turno de cualquier jugador,
            cuando un efecto es activado: puedes seleccionar 1 carta en tu Cementerio; Invócala de Modo Especial.

            - Efecto Disparo: Son efectos que son activadosen momentos específicos, como "durante tu Stanby Phase" o
            "cuando este monstruo es destruido", etc... (más adelante se hablará de Standby Phase y relacionados).

            - Efecto de Volteo: Son parte de los Efectos de Disparo. Estos son activados cuando una carta boca abajo
            se voltea boca arriba. Estos efectos comienzan con la palabra "VOLTEO:" en la carta. Ejemplo: 
            VOLTEO: Roba 1 carta.
        
        Ejemplo de Monstruo de Efecto:""")
    
    st.image(Image.open("./imagenes_videos/Carta_Monstruo_Efecto.png"),caption="Monstruo de Efecto con todas las partes de una Carta de Monstruo")
    
    st.markdown("""
        - Monstruos de Ritual: Estos monstruos NO PUEDEN SER INVOCADOS DE MODO NORMAL O TRIBUTO, necesariamente tienen
        que ser invocados de modo especial  desde la mano por su propia carta de ritual o por otros efectos que lo 
        permitan. Para saber si un monstruo es de ritual solo basta con ver la línea donde está el Tipo ahí mismo se
        observa un apartado que dice "/RITUAL". Los monstruos que se usan como tributo para la invocación de los monstruos
        de Ritual pueden estar boca a bajo y aún se pueden usar como tributo. Estas cartas van en el Main Deck. Ejemplo de invocación por ritual:""")

    rt_sm = open("./imagenes_videos/ritual_invocacion.gif","rb")
    cont_rt = rt_sm.read()
    url_rt = base64.b64encode(cont_rt).decode("utf-8")
    rt_sm.close()

    st.markdown(f'<img src="data:image/gif;base64,{url_rt}" alt="Fusion Invocacion">',
                unsafe_allow_html=True)
    

    st.markdown("""
        - Monstruos de Fusion: Estos monstruos solo pueden ser invocados de modo especial usando la carta 'Polimerización' o
        'Fusión', usando como materiales de fusión monstruos que estén en el campo o en la mano (a menos que la carta permita
        que pueda ser invocado de otras maneras). Los monstruos pueden estar boca abajo y aún podrían ser usados como material
        de fusión. Estas cartas van en el Extra Deck. Ejemplo de invocación por fusión:""")

    fs_sm = open("./imagenes_videos/Fusion_Invocacion.gif","rb")
    cont_fs = fs_sm.read()
    url_fs = base64.b64encode(cont_fs).decode("utf-8")
    fs_sm.close()

    st.markdown(f'<img src="data:image/gif;base64,{url_fs}" alt="Fusion Invocacion">',
                unsafe_allow_html=True)
    
    st.markdown("""
        - Monstruos de Sincronía: Para la invocación de estos monstruos es necesario tener en el campo un monstruo de Tipo 'cantante'
        y monstruos no cantantes cuya suma total de los niveles de los monstruos den exactamente el Monstruo de Sincronía a invocar y
        que a su vez, cumplan los requisitos del Monstruo de Sincronía. Los monstruos que se usarán como materiales para la invocación
        por sincronía deben estar boca arriba para ser utilizados como material. Los textos que tengan cosas como '2+' quiere decir
        '2 o más'. Estas cartas van en el Extra Deck. \n Ejemplo de invocación por sincronía: """)

    sc_sm = open("./imagenes_videos/Sincro_invocacion.gif","rb")
    cont_sc = sc_sm.read()
    url_sc = base64.b64encode(cont_sc).decode("utf-8")
    sc_sm.close()

    st.markdown(f'<img src="data:image/gif;base64,{url_sc}" alt="Sincro Invocacion">',
                unsafe_allow_html=True)

    st.markdown("""- Monstruos XYZ: Para la invocación de estos monstruos es necesario tener 2 o más monstruos DEL MISMO NIVEL (dependiendo de los
    requisitos del monstruo XYZ) y sobre poner un monstruo encima del otro para después colocar desde el Extra Deck al monstruo
    XYZ. Los monstruos que se han sobre puesto YA NO SON CARTAS DE MONSTRUOS, ahora se tratan como materiales XYZ. El oponente no puede
    interactuar con los Materiales a menos que una carta lo permita. Los monstruos que se usarán para traer al monstruo XYZ deben de
    estar boca arriba para poder invocar al monstruos XYZ. Los monstruos XYZ NO TIENEN NIVEL tiene RANGOS, por lo tanto no pueden
    ser usados como material de sincronía ya que no tienen nivel. Estas cartas van en el Extra Deck. \n Ejemplo de invocación por XYZ:
    """
    )

    xyz_sm = open("./imagenes_videos/XYZ_invocacion.gif","rb")
    cont_xyz = xyz_sm.read()
    url_xyz = base64.b64encode(cont_xyz).decode("utf-8")
    xyz_sm.close()

    st.markdown(f'<img src="data:image/gif;base64,{url_xyz}" alt="XYZ Invocacion">',
                unsafe_allow_html=True)
    
    st.markdown("*Todos los videos anteriores son de PlayMaker Duel Links*")
    
    st.subheader("Cartas de Magia")
    st.write(
        """
        Las cartas Mágicas son cartas que se activan en las columnas 2 a 4 segunda fila. Estas cartas tienen la peculiaridad que todas se
        pueden activar en el mismo turno que esa carta llegó a tu mano. Estas cartas van en el Main Deck Existen 5 tipos de cartas de Magia.
        """
    )

    st.markdown(
        """
        - Mágicas Normales: Cartas de un solo uso, este tipo de magia solo puede activarse durante tu turno a menos que el efecto de otra carta
        permita que la puedas activar durante el turno del rival. Estas cartas de magia también pueden activarse en el turno que son colocadas.
        """
    )

    st.markdown(
        """
        - Mágicas Continuas: Estas cartas permanecen en el Campo una vez que han sido activadas y su efeto continúa mientras la carta permanezca
        boca arriba en el Campo. Esta carta de magia también puede activarse en el turno que son colocadas.
        """
    )

    st.markdown(
        """
        - Mágicas de Equipo: Estas cartas permancen boca arriba una vez que han sido activadas, a diferencia de las mágicas continuas. Es que
        estas cartas solo afectan al monstruo que tiene equipado esta carta. Si el monstruo equipado deja el campo, entonces destruye todas 
        las cartas de equipo que estuvieran equipadas a ese monstruo.
        """
    )

    st.markdown(
        """
        - Mágicas de Campo: Esta carta es colocada en la primer columna primer fila. Este tipo de magias a diferencia de todas las demás es que
        casi todas las mágicas de campo afectan a ambos jugadores al mismo tiempo. Cada jugador solo puede tener 1 carta mágica de campo a la vez,
        para activar otra carta de campo es necesario mandar al cementerio la que antes ocupaba ese lugar.
        """
    )

    st.markdown(
        """
        - Mágicas de Juego Rápido: Estas cartas son parecidas a las mágicas normales, solo que a diferencia de las normales, estas se pueden 
        activar en cual quier 'Phase' de tu turno y también se pueden activar durante el turno de tu adversario, pero hay que tener cuidado con
        el Damage Step ya que casi no hay cartas que puedan activar sus efectos en esa fase del duelo. También, no puedes activar este tipo de
        mágicas en el turno que son colocadas.
        """
    )

    st.subheader("Cartas de Trampa")
    st.write(
        """
        Las cartas de trampa a diferencia de las mágicas, NO PUEDEN ACTIVARSE EN EL TURNO QUE SON COLOCADAS, tampoco puedes activarlas en el turno
        que tienes esa carta de trampa en la mano (a menos que la propia carta lo permita). Todas deben primero ser colocadas en el campo.
        Estas cartas van en el Main Deck. Existen 3 tipos de cartas de trampa:
        """
    )

    st.markdown(
        """
        - Trampa Normal: Tienen efectos de un solo uso y una vez que ya fueron resultas son mandadas al cementerio (como las mágicas normales),
        la diferencia es que estas cartas las puedes activar en cualquier momento después del turno que fueron colocadas.
        """
    )

    st.markdown(
        """
        - Trampa Continuas: Son iguales a las mágicas continuas, la única diferencia es que hay mejores efectos en las cartas de trampa continua
        que en las de magia continua.
        """
    )

    st.markdown(
        """
        - Trampa de Contra Efecto: Son cartas que generalmente se activan en respuesta a la activación de otras cartas, y pueden negar efectos
        de esas cartas. El problema es que casi siempre te piden un coste de activación dichas cartas de trampa de contra efecto. Pero la gran
        ventaja que tienen estas cartas por encima de todas las demás (incluyendo magia de juego rápido) es que solamente se le puede responder
        con otra carta de contra efecto.
        """
    )

    st.subheader("Reglas Básicas del Juego")
    st.write(
        """
        En esta parte solo hablaré las reglas más simples y fáciles de entender para que puedas comenzar a jugar, pero se recomienda leer todo
        el reglamento del juego (https://img.yugioh-card.com/es/rulebook/es.pdf). Ambos jugadores empiezan con 4000 puntos de vida (también se
        puede escribir como LP, de hecho esa es la escritura que usa el juego). Tu deck debe tener como mínimo 20 cartas y como máximo 30. El 
        Extra Deck a lo más puede tener 7 cartas. Se lanza una moneda para definir quien empieza. Ambos jugadores empiezan con 4 cartas en la mano. 
        Durante cada turno debes robar 1 carta (excepto el primer turno del jugador que sacó el lanzamiento de moneda). Puedes usar la habilidad que
        quieras. Gana el jugador que logre reducir los LP de su adversario a 0, que el adversario no pueda robar una carta cuando debe hacerlo y
        ganar por el efecto especial de una carta (exodia por ejemplo). Bueno por ahora eso es todo lo que necesitas saber para empezar a jugar
        mas sin embargo, nuevamente recomiendo fuertemente leer el reglamento para conocer detalles pequeños que me salté por que no son muy
        relevantes. Si has llegado hasta aquí recomiendo pasar a la pestaña de 'Rullings' para conocer unos cuantos conceptos avanzados que se
        comentan en este proyecto.
        """
    )

elif choise == "Rullings":
    st.title("Conceptos y Reglas Avanzadas")

    st.subheader("Recomendaciones")
    st.write("Antes de seguir, se recomienda haber leido la pesata 'YuGi-Oh Duel Links' para comprender mejor el juego y así entender los conceptos que se verán aquí.")

    st.subheader("Phases del juego")
    st.write(
        """
        Previamente ya se había explicado todo el resto del juego excepto como se comportan las Phases del juego, pues bueno aqui esplicaré
        por encima cada uno.
        """)

    #basewidth = 300
    img_ph = Image.open('./imagenes_videos/Phases.png')
    #wpercent = (basewidth/float(img_nm.size[0]))
    #hsize = int((float(img_nm.size[1])*float(wpercent)))
    #img_nm = img_nm.resize((basewidth,hsize), Image.ANTIALIAS)
    st.image(img_ph,caption="Phases de YuGi-Oh Duel Links")
    
    st.markdown(
        """
        - Draw Phase: El jugador del turno debe robar una carta de su deck

        - Standby Phase: Activanciones de efecto que especifícan esta Phase

        - Main Phase 1: El jugador de turno puede hacer sus jugadas (invocar, activar, etc...)

        - Battle Phase: Momento donde se llevan a cabo las batallas entre los monstruos, esta fase se divide en 4:
            - Star Step: Comienzo de la battle phase
            - Battle Step: Las batallaes entre los monstruos
            - Damage Step: Calculos de daño
            - End Step: Fin de la Battle Phase

        - End Phase: Fin del turno

        Cabe recalcar que cartas que alteren el ATK o DEF de los monstruos, contra efecto, efectos de volteo y cartas que 
        digan efecto rápido y nieguen activaciones serán las únicas que se pueden activar durante el Damage Step
        """
    )

    st.subheader("Prioridad de Turno")
    st.write(
        """
        La prioridad de turno ayuda a saber quien activará sus efectos primero. Eso quiere decir que el jugador de turno siempre
        tendrá la prioridad 1 al activar sus cartas. Mientras que el oponente solo podrá activar cartas en respuesta a la activaciones
        del jugador de turno (a menos que sean Efectos Rápido y que la carta lo pemita).
        """
    )

    st.subheader("PSCT")
    st.write(
        """
        El PSCT o Problem-Solving Card Text, es método de escritura que facilita la lectura y compresión de todas las cartas. En este caso
        se hablará de la 'condición de activación', 'coste de activación' y el 'efecto'.
        """
    )

    st.markdown(
        """
        - Condicion de Activación: Estos son los requisitos que se necesitan para poder activar las cartas. Las condiciones de activación 
        pueden ser identificados por su requisitos seguido de ':'. Estas condiciones pueden ser de 2 tipos, opcionales u obligatorio.
        Los opcionales dan la oportunidad al controlador de la carta el querar activar el efecto de la carta, normalmente con textos de
        'puedes'. Mientras que los obligatorios ocurren en el momento que se cumple su condición de activación.
        
        Por ejemplo Renunkuriboh, donde vemos que la condición de activación es 'Si esta carta es Sacrificada: ...' Todo lo que esté antes de
        los ':' es la condición de activación. Y esta condición de activación es obligatoria, ya que el robar 1 carta no es opcional, no hay un
        texto que dice 'puedes robar 1 carta'.
        """
    )

    basewidth = 300
    img_cond = Image.open('./imagenes_videos/condicion.jpg')
    wpercent = (basewidth/float(img_cond.size[0]))
    hsize = int((float(img_cond.size[1])*float(wpercent)))
    img_cond = img_cond.resize((basewidth,hsize), Image.ANTIALIAS)
    st.image(img_cond,caption="Ejemplo de la condición de activación")


    st.markdown(
        """
        Ejemplo de Condición de activación opcional, Stratos: la condición de activación de estratos es 'Cuando esta carta es Invocada de Modo
        Normal o Especial: ...' Ahora aquí es donde llega la condicion opcional, '**puedes** activar 1 de estos efecto.'
        """
    )


    img_cond_obl = Image.open('./imagenes_videos/condicion_obl.jpg')
    wpercent = (basewidth/float(img_cond_obl.size[0]))
    hsize = int((float(img_cond_obl.size[1])*float(wpercent)))
    img_cond = img_cond_obl.resize((basewidth,hsize), Image.ANTIALIAS)
    st.image(img_cond_obl,caption="Condición de activación opcional")

    st.markdown(
        """
        - Coste de activación: Los costes de activación se pueden identificar por su coste seguidos por ';'. Estos costes normalmente pueden
        'selecionar' o 'descartar'. 

        Por ejemplo Rompe Raigeki donde se puede observar que todo el coste de activación es 'Descarta 1 carta, y después selecciona 1 carta
        en el Campo; ...' y el efecto es 'destrúyela'
        """
    )

    img_cs = Image.open('./imagenes_videos/coste.jpg')
    wpercent = (basewidth/float(img_cs.size[0]))
    hsize = int((float(img_cs.size[1])*float(wpercent)))
    img_cs = img_cs.resize((basewidth,hsize), Image.ANTIALIAS)
    st.image(img_cs,caption="Coste de activación")


    st.subheader("Cadena")
    st.write(
        """
        Las denas son la forma en cual se decidirá cuáles cartas se resolveran primero y cuáles hasta el final. Si el jugador de turno activa
        un efecto de una carta, su rival SIEMPRE tiene la oportunidad de responder generando así una cadena donde la última respuesta a activación
        será la primera en resolverse y la primera en el eslabón de la cadena será la última en resolverse 
        """
    )

    img_cd = Image.open('./imagenes_videos/cadena.jpg')
    st.image(img_cd,caption="Cadenas y sus Eslabón")

    st.markdown(
        """
        ## Ejemplo

        Supongamos que en algún momento de la partida el jugador 1 activa la Carta Mágica 'Dark Hole' (Efecto de Dark Hole: Destruye todos los
        monstruos en el Campo), entonces nuestro rival responde a la activación de 'Dark Hole' con la Contra Trampa 'Magic Jammer' (Efecto
        de Magic Jammer: Cuando una Carta Mágica es activada: descarta 1 carta; niega la activación y, si lo haces, destrúyela). Por lo que,
        cadena 1 es la Carta Mágica 'Dark Hole' y cadena 2 es la Trampa de Contra Efecto 'Magic Jammer', las reglas de las cadenas nos dicen
        que primero se resulve 'Magic Jammer', por lo que nuestro oponente descarta 1 carta y se activa el efecto negando la activación
        de la Carta Mágica 'Dark Hole', se completa eslabón 2 de la cadena y se pasa a eslabón 1 de la cadena. 'Dark Hole' querrá activar su
        efecto pero este se verá negado por 'Magi Jammer', no se activará y es mandada al cementerio. Fin de la cadena.
        """
    )
    img_cl = Image.open('./imagenes_videos/chain_link.jpg')
    st.image(img_cl,caption="Ejemplo de Cadenas")

    st.subheader("Miss Timing / Perder el tiempo de activación")
    st.write(
        """
        Este es un apartado le cual hoy en día es necesario saber si o si ya que puede ser usado a tu favor. El perder el tiempo de activación
        ocurre cuando cuando se quiere activar el efecto opcional de una carta pero debido a que otros sucesos en el juego ocurren en el momento,
        impiden que ese efecto opcional sea activado. Se van a dar 2 ejemplos, 1 con imagenes el otro con video. Estos efectos son causado por
        las clausulas 'si' y 'cuando', también por las clausulas 'entonces' y 'y si lo haces'.
        """
    )

    st.markdown(
        """
        Caso de perder tiempo de activación con clausulas **if** y **cuando**

        Desde un principio es necesario aclarar que efectos de 'Si esta carta es invocada' y 'Cuando esta carta es Invocada' NO SON LO MISMO.
        La diferencia cae en que **si** es una condición mientras que **cuando** es un periodo de tiempo. Por lo que las cartas que tienen
        clausulas **cuando** pierden su tiempo de activación en comparación de las clausulas **si** que esas nunca pierden su tiempo de 
        activación.

        Ejemplo con goblindbergh y Stratos:

        - Efecto de goblindbergh: Cuando esta carta es Invocada de Modo Normal: puedes Invocar de Modo Especial, desde tu mano, 1 monstruo de Nivel 4 o menor, y además, después de eso, cambiar esta carta a Posición de Defensa.

        - Efecto de hereo elemental stratos: Cuando esta carta es Invocada de Modo Normal o Especial: puedes activar 1 de estos efectos. ●Destruye hasta tantas Mágicas/Trampas en el Campo como la cantidad de monstruos "HÉROE" que controles, excepto esta carta. ●Añade a tu mano 1 monstruo "HÉROE" en tu Deck.


        Primero: Supongamos que tenemos en nuestra mano a estos dos monstruos.
        """
    )

    img_msst = Image.open('./imagenes_videos/ejemplo_paso_1.png')
    st.image(img_msst)

    st.markdown(
        """
        E invocamos de modo normal a Goblindbergh.
        Como goblindbergh fue invocado de Modo Normal entonces podemos Invocar de Modo especial un monstruo de nivel 4 o menor en nuestra mano,
        en esta caso Stratos. Y justo ahora Stratos tiene que activar su efecto pero si recuerdan el efecto de Goblindbergh es que después de
        que este haya Invocado de Modo Especial 1 monstruo de la mano, Goblindbergh tiene que pasar a posición de defensa, pero eso aún no ha
        ocurrido.
        """
    )

    img_msst = Image.open('./imagenes_videos/ejemplo_paso_2.png')
    st.image(img_msst)

    st.markdown(
        """
        Por lo que Goblindbergh pasa a posición de defensa y automáticamente HEROE Elemental Stratos ha perdido su tiempo de activación
        """
    )

    img_msst = Image.open('./imagenes_videos/ejemplo_paso_3.png')
    st.image(img_msst)

    st.markdown(
        """
        ¿Por qué? Por que el momento de activación de del efecto de Stratos es JUSTO CUANDO TOCA EL CAMPO. No antes, no después. Pero
        ¿Qué fue lo que pasó en el momento de la invocación de Stratos? Lo último que se realizó en el campo fue que Goblindbergh pasó
        a defensa y no la activación de Stratos. Debido a que Stratos tiene un **cuando** significando periodo de tiempo.

        Ahora cambiemos el ejemplo de Stratos a Oviraptor Devorando Almas.

        - Efecto de Oviraptor Devorando Almas: Si esta carta es Invocada de Modo Normal o Especial: puedes tomar 1 monstruo de Tipo Dinosaurio en tu Deck y añádelo a tu mano o mándalo al Cementerio. Puedes seleccionar otro monstruo de Tipo Dinosaurio de Nivel 4 o menor en el Campo; destrúyelo, y después Invoca de Modo Especial en, Posición Defensa, 1 monstruo de Tipo Dinosaurio en tu Cementerio. Sólo puedes usar cada efecto de "Oviraptor Devorando Almas" una vez por turno.

        Primero: Supongamos que tenemos en nuestra mano a estos dos monstruos.
        """
    )

    img_msst = Image.open('./imagenes_videos/ejemplo2_paso_1.png')
    st.image(img_msst)

    st.markdown(
        """
        E invocamos de modo normal a Goblindbergh.
        Como goblindbergh fue invocado de Modo Normal entonces podemos Invocar de Modo especial un monstruo de nivel 4 o menor en nuestra mano,
        en esta caso Oviraptor. 
        """
    )

    img_msst = Image.open('./imagenes_videos/ejemplo2_paso_2.png')
    st.image(img_msst)

    st.markdown(
        """
        Y a difernecia de Stratos, Oviraptor no tiene que activar su efecto justo ahora por lo que el efecto de Goblindbergh
        de tener que pasar a posición de defensa ocurre. Hasta entonces ahora si Oviraptor puede activar su efecto, ya que se ha 
        cumplido la condición de Oviraptor, fue Invocado de Modo Especial, y ya con eso basta para que se pueda activar su efecto. 
        No es necesario que se active justo cuando toca el campo.
        """
    )

    img_msst = Image.open('./imagenes_videos/ejemplo2_paso_3.png')
    st.image(img_msst)

    st.markdown(
        """
        ## Pregunta

        ¿El Sabio Oscuro pierde tiempo de activación?
    """)
    
    img_tst = Image.open('./imagenes_videos/sabio_oscuro.jpg')
    wpercent = (basewidth/float(img_tst.size[0]))
    hsize = int((float(img_tst.size[1])*float(wpercent)))
    img_tst = img_tst.resize((basewidth,hsize), Image.ANTIALIAS)
    st.image(img_tst)

    st.markdown("""
        Efecto de Sabio Oscuro: No puede ser Invocado de Modo Normal, ni Colocado. Debe ser primero Invocado de Modo Especial (desde tu mano o Deck) Sacrificando 1 "Mago Oscuro" inmediatamente después de aplicar el efecto de "Mago del Tiempo" en el que acertaste en el lanzamiento de la moneda. Cuando este monstruo es Invocado de Modo Especial de esta forma: añade a tu mano, desde tu Deck, 1 Carta Mágica.

        ## Respuesta: 
        **NO**, aun que pareciera que si puede perder tiempo de activación la realidad es que ese 'cuando' es mandatorio o como previamente se
        explicó, es un efecto obligatorio. La carta no pregunta si quieres agregar 1 Carta Mágica a tu mano, te obliga a tener que agregarte
        una Carta Mágica, y en caso de no tener no agregas nada.

        Ahora se usará un video como ejemplo para que se pueda observar en tiempo real que ocurre. En este caso tenemos a la carta de campo
        'Pueblo Mecánico' y la Carta Mágica 'Tormenta'.

        - Efecto de Pueblo Mecánico: Ambos jugadores pueden Invocar de Modo Normal monstruos "Mecanismo Antiguo" haciendo 1 Sacrificio menos. Cuando esta carta es destruida y mandada al Cementerio: puedes Invocar de Modo Especial, desde tu mano, Deck o Cementerio, 1 monstruo "Mecanismo Antiguo".

        - Efecto de Tormenta: Destruye tantas Mágicas/Trampas que controles como sea posible, y después destruye tantas Mágicas/Trampas que controle tu adversario como sea posible, hasta la cantidad de cartas destruidas por este efecto.

        Lo que esta pasando es que cuando se activa el efecto de 'Tormenta' **primero** se tiene que destruir tus cartas y **después** las de tu
        oponente y aqui es donde llega el problema, cuando tu destruyas tus cartas, entre ellas 'Pueblo Mecánico' se va a querer activar el efecto
        de 'Pueblo Mecánico' pero va a ser imposible por que justo en ese momento se tienen que destruir las cartas del oponente. Haciendo que 
        'Pueblo Mecánico' pierda su tiempo de activación.
        """
    )

    miss_tm = open("./imagenes_videos/miss_timing.gif","rb")
    cont_tm = miss_tm.read()
    url_tm = base64.b64encode(cont_tm).decode("utf-8")
    miss_tm.close()

    st.markdown(f'<img src="data:image/gif;base64,{url_tm}" alt="Miss Timing">',
                unsafe_allow_html=True)
    
    st.markdown("*Video de Veiz")

elif choise == "Herramientas":

    st.title("Herramientas")
    st.write(
        """
        Este apartado solo es para mencionar funciones de los archivos .py que se encuentra en la carpeta 'herramientas' de este proyecto.
        Se recomienda haber estado en 'YuGi-Oh Duel Links' y 'Rullings' para entender mejor algunos conceptos hacer estas funciones.
        \n
        En esta carpeta se encuentran 2 archivos 'card_filter' y 'to_basic'.
        """
    )

    st.subheader("card_filter")
    
    st.markdown(
        """
        La ultilidad que tiene este archivo es para la busqueda de las cartas por similitud y para el conteo de palabras.
        """
    )

    with st.echo():
        def similitud_texto(texto_base,texto_entrada,tipo_metodo="smb"):
            """
            Calcula una similitud entre 2 palabras, mientras mas pequeño sea el valor mas similares son

            Parametros
            ----

            texto_base: str, obligatorio. El texto principal y con el cual se haran las comparaciones

            texto_entrada: str, olbigatorio. El texto al cual se quiere saber si es similar al texto base

            tipo_metodo: str, opcional ([smb],sme). La forma con la cual se hara la comparacion, similut basica o similitud extricta

            Regresa
            ----
            Un numero entero de 0 en adelante
            """
        
        def buscar_nombre(nombre,data,tipo_busqueda="ex",epsilon=None):
            """
            Regresa una o varias cartas con el nombre ingresado dependiendo del tipo de busqueda que se haga

            Parametros
            ----

            nombre: str, obligatorio. El nombre de la carta a buscar
            
            data: pd.DataFrame, obligatorio. El data set en el que se buscara la carta

            tipo_busqueda: str, opcional. El metodo de busqueda
                ex: Se buscara exactamente como este nombre
                smb: Buscara una similitud basica con el nombre, donde la posicion de las letras **no importa**, si epsilon != None, entonces se buscara todos los resultados menores a ese epsilon
                sme: Ahora es una similitud estricta donde **si importa** la posicion de las letra, si epsilon != None, entonces se buscara todos los resultados menores a ese epsilon
                lv: Usara el algoritmo de Levenshtein para ello es necesario ingresar un 'epsilon' como parametro de la funcion
            
            Regresa
            ----
            pd.DataFrame con las coincidencias encontradas
                    
            """
        
        def contar_pal(data,columna="Texto de la Carta",PSCT=False,axis=1):
            """
            Cuenta cuantas veces aparece cada palabra en la columna del Data Set

            Parametros
            ----

            data: pd.DataFrame, obligatorio. El data set al que se le aplicara el conteo

            columna: [str],list, opcional. La o las columnas a las que se le aplicara el conteo

            PSCT: bool, opcional. Contará de diferente manera las palabras que su estructura dependa del PSCT, ejemplo PSCT=True:
                                    \... descarta 1 carta; .... <-- "descarta 1 carta" es el coste
                                    \... ; descarta 1 carta ... <-- "descarta 1 carta" es el efecto
                                    La palabra 'descarta' sera contada como 2 tipos diferentes de palabras

            axis: int[0,1]. En que formato regresara el dataframe, 0: filas, 1: columnas
            
            Regresa
            ----

            Si PSCT = False: pd.DataFrame de tamaño (1, palabras diferentes)
            Si PSCT = True: pd.DataFrame de tamaño(2, palabra diferentes) 
            """
        
        def buscar_carta_por_especificacion(data,**kwargs):
            """
            Buscara las cartas que coincidan dadas las especificaciones pasadas como parametros

            Parametros
            ----

            data: pd.DataFrame, obligatorio. El dataframe al cual se le aplicara la busqueda

            **kwargs: obligatorio (min 1). Nombre de la columna y condicion de busqueda

            Regresa
            ----
            pd.DataFrame con todas las coincidencias

            Ejemplo
            ----

            >>> path_actual = str(os.getcwd()).split("\\")
            >>> path_db = str(os.getcwd())[:-12]+"base_datos\listacartas_basedatos\\"
            >>> data = pd.read_csv(path_db+"CardList_DataBase.csv")[:-1]
            >>> buscar_carta_por_espeficicacion(data,Método_de_Obtención="Gratis",Texto_de_la_Carta="#")
               Tipo de Carta               Nombre de Carta Texto de la Carta Tipo de Magia Tipo de Trampa  ...     ATK     DEF  Fecha de Publicación de la Carta  Rareza  Método de Obtención
            0       Monstruo  Dragon Blanco de Ojos Azules                 #             #              #  ...  3000.0  2500.0                        11/01/2017      UR               Gratis     
            1       Monstruo    Dragon Negro de Ojos Rojos                 #             #              #  ...  2400.0  2000.0                        11/01/2017      UR               Gratis     
            2       Monstruo                   Mago Oscuro                 #             #              #  ...  2500.0  2100.0                        11/01/2017      UR               Gratis     
            84      Monstruo          HEROE Elemental Neos                 #             #              #  ...  2500.0  2000.0                        28/09/2017      UR               Gratis     
            92      Monstruo       HEROE Elemental Clayman                 #             #              #  ...   800.0  2000.0                        28/09/2017       R               Gratis     

            [5 rows x 13 columns]
            """
        
    st.subheader("to_basic")

    with st.echo():
        def to_minus(data,columnas=None,acentos=True,comas="Nombre de Carta"):
            """
            Convierte todas las letras del data set de todas las columnas seleccionadas a minusculas

            Parametros
            -----
            data: pd.DataFrame, obligatorio. El data al cual se le aplicará esto

            columnas: str,list, opcional. Cuales columnas se les aplicara minus, si es None a todas

            acentos: bool, opcional. True quitara todos los acentos del data set

            comas: str,list, opcional: El nombre de la o las columnas a las que se les quitara la coma

            Regresa
            ----
            pd.DataFrame orginal pero todas las letras en minusculas
                """


    