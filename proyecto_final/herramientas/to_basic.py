import pandas as pd
import os

def to_minus(data,columnas=None,acentos=True,comas="Nombre de Carta"):
    """
    Convierte todas las letras del data set de todas las columnas seleccionadas a minusculas

    Parametros
    -----
    data: pd.DataFrame, obligatorio. El data al cual se le aplicará esto

    columnas: str,list, opcional. Cuales columnas se les aplicara minus, si es None a todas

    acentos: bool, opcional. True quitara todos los acentos del data set

    Regresa
    ----
    pd.DataFrame orginal pero todas las letras en minusculas
    """

    if isinstance(columnas,str):
        columnas = [columnas]
    if isinstance(comas,str):
        comas = [comas]
    
    
    c = ["Tipo de Carta","Nombre de Carta","Texto de la Carta","Tipo de Magia","Tipo de Trampa","Tipo de Monstruo","Atributo","Rareza","Método de Obtención"]
    for i in c if columnas == None else columnas: #columna
        texto = []
        for t in data[i]: #texto
            if acentos:
                aux = ""
                dic = {"á":"a","é":"e","í":"i","ó":"o","ú":"u"}
                for l in t: #letra por letra
                    if i in comas:
                        if l == ",":
                            l = ""
                    if l in dic.keys():
                        aux += dic[l.lower()]
                    else:
                        aux += l.lower()
                texto.append(aux)
            else:
                texto.append(t.lower())
        data[i] = texto
    
    return data
        

    
if "__main__" == __name__:
    path_actual = str(os.getcwd()).split("\\")
    path_db = str(os.getcwd())[:-12]+"base_datos\listacartas_basedatos\\"
    data = pd.read_csv(path_db+"CardList_DataBase.csv")[:-1]
    print(to_minus(data))