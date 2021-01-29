import pandas as pd
import os
import nltk
from copy import copy
import numpy as np

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

    if tipo_metodo == "smb" or tipo_metodo == "sme":
        distancia = 0
        pal1 = texto_base.split()
        pal2 = texto_entrada.split()

        for i,t1 in enumerate(pal1):
            min_distances = None
            for j,t2 in enumerate(pal2):
                #si smb entonces se reemplazaran los caracteres especiales
                if tipo_metodo == "smb":
                    caracteres = {"á":"a","é":"e","í":"i","ó":"o","ú":"u",
                                  "!":"","¡":"","\"":"","#":"","(":"",")":"","'":"",".":"",";":"",":":"","-":""}
                    aux_t1 = ""
                    cont_t1 = 0
                    aux_t2 = ""
                    cont_t2 = 0
                    for at in t1:
                        if at in caracteres.keys():
                            aux_t1 += caracteres[at]
                            try:
                                cont_t1 += ord(caracteres[at])
                            except:
                                cont_t1 += 1
                        else:
                            aux_t1 += at
                    for at in t2:
                        if at in caracteres.keys():
                            aux_t2 += caracteres[at]
                            cont_t2 += 1
                            try:
                                cont_t2 += ord(caracteres[at])
                            except:
                                cont_t2 += 1
                        else:
                            aux_t2 += at
                    t1 = aux_t1
                    t2 = aux_t2

                _val = nltk.edit_distance(t1,t2)
                if j == 0:
                    if tipo_metodo == "smb":
                        min_distances = [j,_val + cont_t1 + cont_t2]
                    else:
                        min_distances = [j,_val]
                else:
                    if _val < min_distances[1]:
                        if tipo_metodo == "smb":
                            min_distances = [j,_val + cont_t1 + cont_t2]
                        else:
                            min_distances = [j,_val]
            if min_distances != None:
                del pal2[min_distances[0]]
                distancia += min_distances[1]

        return distancia
    else:
        raise TypeError("El metodo " + str(tipo_metodo) + " no es reconocible")
        

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
    if tipo_busqueda == "ex":
        return data[data["Nombre de Carta"] == nombre]

    elif tipo_busqueda == "lv":
        return data[data["Nombre de Carta"].apply(lambda e: nltk.edit_distance(e,nombre)) <= epsilon]
    
    elif tipo_busqueda == "smb" or tipo_busqueda == "sme":
        similitudes = []
        for i in data["Nombre de Carta"]:
            similitudes.append(similitud_texto(nombre,i.lower(),tipo_metodo=tipo_busqueda))
        if epsilon == None:
            epsilon = np.mean(similitudes)/np.mean(data["Nombre de Carta"].str.split().apply(len))
            
            aux = copy(data)
            aux["Nombre de Carta"] = similitudes
            
            if tipo_busqueda == "smb":
                return data[(aux["Nombre de Carta"] <= epsilon)] 
            else:
                return data[(aux["Nombre de Carta"] <= epsilon) & (data["Nombre de Carta"].str.split().apply(len) >= len(nombre.split()))]
        else:
            aux = copy(data)
            aux["Nombre de Carta"] = similitudes
            return data[aux["Nombre de Carta"] <= epsilon] 


def buscar_texto(texto,data,tipo_busqueda="ex",epsilon=None):
    if tipo_busqueda == "ex":
        filas = []
        for k,i in enumerate(data["Texto de la Carta"]):
            if texto in i:
                filas.append(k)
        return data.loc[filas]
    elif tipo_busqueda == "smb" or tipo_busqueda == "sme":
        similitudes = []
        for i in data["Texto de la Carta"]:
            similitudes.append(similitud_texto(texto,i.lower(),tipo_metodo=tipo_busqueda))
        if epsilon == None:
            epsilon = np.mean(similitudes)
            aux = copy(data)
            aux["Texto de la Carta"] = np.array(similitudes)           
            return data[(aux["Texto de la Carta"] <= epsilon-min(similitudes)) & (data["Texto de la Carta"] != "#")]
        else:
            aux = copy(data)
            aux["Texto de la Carta"] = np.array(similitudes) 
            return data[(aux["Texto de la Carta"] <= epsilon-min(similitudes)) & (data["Texto de la Carta"] != "#")] 



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
    """

    for i in kwargs.keys():
        kwargs[i.replace("_"," ")] = kwargs.pop(i)
    index = pd.DataFrame(data[key] == val for key,val in kwargs.items()).T.all(axis=1)
    return data[index]

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

    if isinstance(columna,str):
        columna = [columna]
    
    palabras = {}
    new_palabras = {}
    caracteres = {"!":"","¡":"","\"":"","#":"","(":"",")":"","'":"",";":"",":":"","-":""}
    _psct = {"mt":"Mismo Texto","pt":"Aclaracion o Condiciones","vn":"Viñeta","ct":"Coste","cd":"Condicion de Activacion","ef":"Efecto"}
    for n in columna:
        for c in data[n]: #texto
            parentesis = False
            coste = False
            condicion = False
            vinetas = False
            m_texto = False
            punto = []
            condition = []
            #PSCT = True
            aux = ""
            palabras = {}
            for k,l in enumerate(c): #letra
                if PSCT:
                    if l == ",":
                        l = ""

                    if l == "(" or l == ")": #los parentesis no los guardo en punto
                        parentesis = not parentesis
                    elif l == ";":
                        coste = not coste
                        aux += "ct" if len(aux) > 0 else ""
                        for s in punto:
                            palabras[s] -= 1
                            if s[:-2]+"ct" in palabras.keys():
                                palabras[s[:-2]+"ct"] += 1
                            else:
                                palabras[s[:-2]+"ct"] = 1
                        if aux in palabras.keys():
                            palabras[aux] += 1
                        else:
                            if aux != "":
                                palabras[aux] = 1
                        aux = ""
                        punto = []
                    elif l == ":":
                        condicion = not condicion
                        aux += "cd" if len(aux) > 0 else ""
                        for s in punto:
                            palabras[s] -= 1
                            if s[:-2]+"cd" in palabras.keys():
                                palabras[s[:-2]+"cd"] += 1
                            else:
                                palabras[s[:-2]+"cd"] = 1
                        if aux in palabras.keys():
                            palabras[aux] += 1
                        else:
                            palabras[aux] = 1
                        aux = ""
                        punto = []
                    elif l == "-" and c[k-1] == " ":
                        vinetas = not vinetas
                        
                    elif l == "\"": #tampoco los que esten entre comillas
                        m_texto = not m_texto
                    elif l == " " or l == "." or k+1 >= len(c) and len(aux) > 0:
                        if parentesis:
                            aux += "pt" if len(aux) > 0 else "" #pt = parentesis
                        elif vinetas:
                            aux += "vn" if len(aux) > 0 else "" #vn = viñetas
                        elif m_texto:
                            aux += "mt" if len(aux) > 0 else "" #mt = mismo texto
                        else:
                            if c[k-1] != "\"" and c[k-1] != ")":
                                aux += c[k] if k+1 >= len(c) and c[k] != "." else ""
                                aux += "ef" if len(aux) > 0 else "" #ef = efecto
                            

                        if aux in palabras.keys():
                            palabras[aux] += 1
                            punto.append(aux)
                            aux = ""
                        else:
                            if aux != "":
                                if c[k-1] == "\"":
                                    aux += "mt" if len(aux) > 0 else "" #mt = mismo texto
                                elif c[k-1] == ")":
                                    aux += "pt" if len(aux) > 0 else "" #pt = parentesis
                                else:
                                    palabras[aux] = 1
                                    punto.append(aux)
                                    aux = ""
                                if len(aux) > 0:
                                    palabras[aux] = 1
                                    aux = ""
                        
                        if (m_texto == False and c[k-1] != "\"") and (parentesis == False and c[k-1] != ")"):
                            if aux != "":
                                punto.append(aux)
                                aux = ""
                        else:
                            if c[k-1] == "\"":
                                aux += "mt" if len(aux) > 0 else "" #mt = mismo texto
                            elif c[k-1] == ")":
                                aux += "pt" if len(aux) > 0 else "" #pt = parentesis

                            if aux in palabras.keys():
                                palabras[aux] += 1
                                punto.append(aux)
                                aux = ""
                            else:
                                if aux != "":
                                    palabras[aux] = 1
                                    punto.append(aux)
                                    aux = ""
                        if l == ".":
                            punto = []
                            m_texto = False
                            vinetas = False
                    else:
                        aux += l
                else:
                    if l in caracteres.keys():
                        aux += caracteres[l]
                    elif l == " " or l == ".":
                        if aux in palabras.keys():
                            palabras[aux] += 1
                            aux = ""
                        else:
                            if aux != "":
                                palabras[aux] = 1
                                aux = ""
                    else:
                        aux += l
                    
            
            for i in palabras:
                if palabras[i] > 0:
                    new_palabras[i] = [palabras[i]] if PSCT == False else palabras[i]
    
    dataF = {}

    if PSCT:
        aux = ""
        for i in new_palabras:
            aux = i[:-2]+"_"+i[-2:]
            try:
                dataF[aux] = [new_palabras[i],_psct[i[-2:]]]
            except:
                print("exception")
        dataF = pd.DataFrame(dataF) 
        dataF.index = ["#","PSCT"]
        return dataF.T if axis==1 else dataF
    else:
        return pd.DataFrame(new_palabras).T if axis==1 else pd.DataFrame(new_palabras)


if "__main__" == __name__:
    path_actual = str(os.getcwd()).split("\\")
    path_db = str(os.getcwd())[:-12]+"base_datos\listacartas_basedatos\\"
    data = pd.read_csv(path_db+"CardList_DataBase.csv")[:-1]
    #print(buscar_nombre("ojos rojos",data,tipo_busqueda="smb"))
    #print(buscar_texto("\"Ojos rojos\"",data,tipo_busqueda="sme"))
    #print(buscar_carta_por_especificacion(data,Método_de_Obtención="Gratis",Texto_de_la_Carta="#"))
    
    
    


    

    