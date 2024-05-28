
#---------------------------------------------------------------------------------#
# ACP : Analyse de Composantes Principales
#---------------------------------------------------------------------------------#

import streamlit as st
import streamlit.components.v1 as components
import styles_app
# from streamlit_carousel import carousel

import os
import time
import numpy as np
import pandas as pd
import scipy as sc




                ## Statistic Test : Chix-2 test                
                st.markdown(f"<style>{css_styles}</style>", unsafe_allow_html=True)
                st.markdown("""
                <div class="container">
                    <div class="header">Statistic Test : Chix-2 test 💡</div>
                    <div class="content">
                        <ul>
                            <li>
                                consists in transforming related variables (known in statistics as “correlated” variables) into new variables that are decorrelated from one another. These new variables are called “principal components” or principal axes. 
                                It enables us to summarize information by reducing the number of variables.
                            </li>

                        </ul>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                
